"""
Four serving strategies behind one interface, so the benchmark can swap them.
 
    S0  SequentialEngine         one request at a time, KV cache, no batching
    S1  StaticBatchEngine        form a batch, run it to completion, repeat
    S2  ContinuousUnbatched      continuous admission/eviction, decode NOT merged
    S3  ContinuousBatched        your Stage 4 engine (merged, left-padded batch)
 
Interface (all engines):
    engine.add_request(req)      enqueue (sets req.arrival_time)
    engine.step() -> bool        do one unit of work; False if idle
    engine.completed_requests    list of finished Requests
    engine.stats                 dict of engine-level counters (see Stats)
 
"""

import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers.cache_utils import DynamicCache

from run import ContinuousBatchEngine, Request

def _sync(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize()

@dataclass
class Stats:
    rebuild_time: float = 0.0
    extract_time: float = 0.0
    rebuild_count: int = 0
    decode_steps: int = 0
    batch_sizes: list = field(default_factory=list)
    wasted_slot_steps: int = 0

def _timed(fn, stats, time_attr, count_attr=None):
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        setattr(stats, time_attr, getattr(stats, time_attr) + time.perf_counter() - t0)
        if count_attr:
            setattr(stats, count_attr, getattr(stats, count_attr) + 1)

        return out
    return wrapper

def _mark_first_token(engine):
    """Record when each request's first token exists (end of its prefill)."""
    orig = engine.prefill_request

    def wrapper(request):
        out = orig(request)
        _sync(engine.model)
        request.first_token_time = time.perf_counter()
        return out

    engine.prefill_request = wrapper

def _finalize(engine, request):
    request.finish_time = time.perf_counter()
    engine.completed_requests.append(request)
    request.done_event.set()

@torch.no_grad()
def _decode_one(engine, request):
    """
    One unbatched decode step for one request (own KV cache). Mirrors the
    semantics of the batched path: the pending token is recorded, a forward
    pass is run on it, and the next token is chosen greedily. Written here
    (instead of calling run.py's decode_request) so input_ids is always
    shaped [1, 1].
    """
    model = engine.model
    _sync(model)
    out = model(input_ids=request.next_token.view(1,1),
                past_key_values=request.past_key_values, use_cache=True)

    _sync(model)
    tok_id = request.next_token.item()
    request.generated_tokens.append(tok_id)
    request.decode_steps += 1
    request.cache_length += 1
    if tok_id == engine.tokenizer.eos_token_id:
        request.done, request.finish_reason = True, "eos"
    elif request.decode_steps >= request.max_new_tokens:
        request.done, request.finish_reason = True, "length"
    request.past_key_values = out.past_key_values
    request.next_token = out.logits[:, -1, :].argmax(-1)

# ---------------------------------------------------------------- S0
class SequentialEngine(ContinuousBatchEngine):
    """FIFO, one request at a time, per-request KV cache (no batching)"""
    name = "Sequential"

    def __init__(self, model, tokenizer, **_):
        super().__init__(model, tokenizer, max_batch_size=1)
        self.stats = Stats()
        _mark_first_token(self)

    def step(self):
        request = self.waiting_queue.pop()
        if request is None:
            return False

        self.prefill_request(request)
        request.batch_entry_time = time.perf_counter()
        while not request.done:
            _decode_one(self, request)
            self.stats.decode_steps += 1
            self.stats.batch_sizes.append(1)
        _finalize(self, request)
        return True

# ---------------------------------------------------------------- S1
class StaticBatchEngine:
    """
    Take up to `max_batch_size` waiting requests, left-pad, and run them
    together until the LONGEST one finishes. Finished rows keep being
    computed (wasted slots) and every response is returned when the whole
    batch completes. This is the textbook static-batching baseline.
    """
    name = "static"

    def __init__(self, model, tokenizer, max_batch_size=4):
        from run import RequestQueue
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.waiting_queue = RequestQueue()
        self.completed_requests = []
        self.stats = Stats()

    def add_request(self, request):
        request.arrival_time = time.perf_counter()
        self.waiting_queue.add(request)

    def step(self):
        batch = []
        while len(batch) < self.max_batch_size:
            r = self.waiting_queue.pop()
            if r is None:
                break
            batch.append(r)

        if not batch:
            return False
        self._run_batch(batch)
        return True

    @torch.no_grad()
    def _run_batch(self, batch):
        tok, model = self.tokenizer, self.model
        old_side = tok.padding_side
        tok.padding_side = "left"
        enc = tok([r.prompt for r in batch], return_tensors="pt", padding=True).to(model.device)
        tok.padding_side = old_side

        input_ids, attn = enc["input_ids"], enc["attention_mask"]
        position_ids = (attn.cumsum(-1) - 1).clamp(min=0)
        now = time.perf_counter()
        for r in batch:
            r.batch_entry_time = now

        out = model(input_ids=input_ids, attention_mask=attn,
                    position_ids=position_ids, use_cache=True)
        _sync(model)

        t_first = time.perf_counter()
        for r in batch:
            r.first_token_time = t_first

        cache = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1)
        last_pos = position_ids[:, -1:]
        max_steps = max(r.max_new_tokens for r in batch)
        eos = tok.eos_token_id

        for step in range(max_steps):
            self.stats.decode_steps += 1
            self.stats.batch_sizes.append(sum(not r.done for r in batch))
            for i, r in enumerate(batch):
                if r.done:
                    self.stats.wasted_slot_steps += 1
                    continue

                t = next_tok[i].item()
                r.generated_tokens.append(t)
                r.decode_steps += 1
                if t == eos:
                    r.done, r.finish_reason = True, "eos"
                elif r.decode_steps >= r.max_new_tokens:
                    r.done, r.finish_reason = True, "length"

            if all(r.done for r in batch):
                break

            attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=-1)
            last_pos = last_pos + 1
            out = model(input_ids=next_tok.unsqueeze(-1), attention_mask=attn,
                        position_ids=last_pos, past_key_values=cache, use_cahe=True)
            cache = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1)

        _sync(model)
        end = time.perf_counter()
        for r in batch:
            r.finish_time = end
            self.completed_requests.append(r)
            r.done_event.set()

# ---------------------------------------------------------------- S2
class ContinuousUnbatchedEngine(ContinuousBatchEngine):
    """Continuous admission/eviction, but one forward pass PER request per
    step. Isolates the effect of scheduling from the effect of batched compute."""

    name = "continuous_unbatched"

    def __init__(self, model, tokenizer, max_batch_size=4):
        super().__init__(model, tokenizer, max_batch_size=max_batch_size)
        self.stats = Stats()
        _mark_first_token(self)

    def step(self):
        if self.waiting_queue.empty() and self.active_batch.empty:
            return False

        self.admit_requests()
        live = [r for r in self.active_batch.requests if not r.done]
        self.stats.decode_steps += 1
        self.stats.batch_sizes.append(len(live))
        for r in live:
            _decode_one(self, r)
        self.finish_requests()
        return True

# ---------------------------------------------------------------- S3
class ContinuousBatchedEngine(ContinuousBatchEngine):
    """Your Stage 4 engine: batched decode over a merged, left-padded cache."""

    name = "continuous_batched"

    def __init__(self, model, tokenizer, max_batch_size=4):
        super().__init__(model, tokenizer, max_batch_size=max_batch_size)
        self.stats = Stats()
        _mark_first_token(self)
        cm = self.cache_manager
        cm.build_batched_cache = _timed(cm.build_batched_cache, self.stats,
                                        "rebuild_time", "rebuild_count")

        cm.extract_single_request_cache = _timed(cm.extract_single_request_cache,
                                                 self.stats, "extract_time")

    def step(self):
        if self.waiting_queue.empty() and self.active_batch.empty:
            return False

        self.admit_requests()
        self.stats.decode_steps += 1
        self.stats.batch_sizes.append(len(self.active_batch))
        self.decode_active_batch()
        self.finish_requests()
        return True

ENGINES = {
    "sequential": SequentialEngine,
    "static": StaticBatchEngine,
    "continuous_unbatched": ContinuousUnbatchedEngine,
    "continuous_batched": ContinuousBatchedEngine,
}