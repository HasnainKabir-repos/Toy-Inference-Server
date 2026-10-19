"""
Controlled benchmark of sequential / static / continuous batching.
 
Examples
    # fast sanity check on a tiny random model (no downloads, runs anywhere)
    python bench/bench.py --model smoke --quick
 
    # core study on the real model
    python bench/bench.py --model fp16 --workloads uniform,mixed --caps 4,8 \
        --repeats 5 --n 24 --out results/cpu.jsonl
 
    # full matrix (use the T4)
    python bench/bench.py --model fp16 --workloads uniform,mixed,longctx \
        --caps 2,4,8 --repeats 5 --n 32 --out results/t4.jsonl
 
Design
  * Every system sees the identical request stream (same seed per repeat).
  * "offline": all requests are present at t=0.
  * "poisson@x": open-loop arrivals at x * (measured sequential capacity).
    Arrivals follow their schedule regardless of server state, and latency is
    measured from the *scheduled* arrival, so slow responses cannot suppress
    load (no coordinated omission).
  * Run order within a repeat is shuffled to spread drift across systems.
  * Greedy decoding; warmup before measurement; one JSON line per run.
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engines import ENGINES, Stats
from workloads import make_workload, poisson_offsets
from run import Request

# ------------------------------------------------------------------ models
class _FakeBatch(dict):
    def to(self, device):
        return _FakeBatch({k: v.to(device) for k, v in self.items()})

class FakeTokenizer:
    """Byte-ish tokenizer for the smoke test only (no network needed)."""
    eos_token_id = 1
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, texts, return_tensors="pt", padding=False):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        ids = [[(ord(c) % 120) + 3 for c in t][:60] for t in texts]
        width = max(len(i) for i in ids)
        rows, masks = [], []
        for i in ids:
            pad = [self.pad_token_id] * (width - len(i))
            if self.padding_side == "left":
                rows.append(pad + i); masks.append([0] * len(pad) + [1] * len(i))
            else:
                rows.append(i + pad); masks.append([1] * len(i) + [0] * len(pad))

        return _FakeBatch(input_ids= torch.tensor(rows), attention_mask=torch.tensor(masks))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def load_model(kind):
    if kind == "smoke":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=16,
                          max_position_embeddings=1024)
        model = Qwen3ForCausalLM(cfg).eval()
        return FakeTokenizer(), model

    import run
    if kind == "fp16":
        return run.load_fp16_model()
    if kind == "nf4":
        return run.load_quantized_model()
    raise ValueError(kind)

def warmup(model, tok):
    for name, cls in ENGINES.items():
        eng = cls(model, tok, max_batch_size=2) if name != "sequential" else cls(model, tok)
        for i in range(3):
            eng.add_request(Request(request_id=i, prompt="Warm up the model.", max_new_tokens=4))
        while eng.step() or len(eng.completed_requests) < 3:
            pass

# ------------------------------------------------------------------ one run
def run_once(system, model, tok, workload, cap, offsets, seed):
    cls = ENGINES[system]
    eng = cls(model, tok) if system == "sequential" else cls(model, tok, max_batch_size=cap)
    n = len(workload)
    reqs = [Request(request_id=i, prompt=p, max_new_tokens=m) for i, (p, m) in enumerate(workload)]

    if model.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    if offsets is None:
        for r in reqs:
            eng.add_request(r)
            r.arrival_time = t0
        feeder = None

    else:
        def feed():
            for r, off in zip(reqs, offsets):
                delay = t0 + off - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                eng.add_request(r)
                r.arrival_time = t0 + off
        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()

    while len(eng.completed_requests) < n:
        if not eng.step():
            time.sleep(0.0005)

    if feeder:
        feeder.join()
    wall = max(r.finish_time for r in reqs) - t0

    per_req = []
    for r in reqs:
        k = r.generated_token_count
        e2e = r.finish_time - r.arrival_time
        ttft = r.first_token_time - r.arrival_time
        tpot = (r.finish_time - r.first_token_time) / (k - 1) if k > 1 else None
        per_req.append(dict(id=r.request_id, tokens=k, budget=r.max_new_tokens,
                            reason=r.finish_reason, ttft=ttft, e2e=e2e, tpot=tpot,
                            queue_wait=(r.batch_entry_time - r.arrival_time),
                            out=r.generated_tokens))
    total_tokens = sum(p["tokens"] for p in per_req)
    s: Stats = eng.stats
    bs = s.batch_sizes
    return dict(
        wall=wall, total_tokens=total_tokens,
        tok_per_s=total_tokens / wall, req_per_s=n / wall,
        mean_occupancy=(sum(bs) / len(bs)) if bs else None,
        decode_steps=s.decode_steps,
        rebuild_time=s.rebuild_time, extract_time=s.extract_time,
        rebuild_count=s.rebuild_count,
        rebuild_frac=(s.rebuild_time + s.extract_time) / wall,
        wasted_slot_steps=s.wasted_slot_steps,
        per_request=per_req,
    )

# ------------------------------------------------------------------ meta
def meta(args):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    import transformers
    m = dict(torch=torch.__version__, transformers=transformers.__version__,
             python=platform.python_version(), platform=platform.platform(),
             threads=torch.get_num_threads(), commit=commit, args=vars(args))
    if torch.cuda.is_available():
        m["gpu"] = torch.cuda.get_device_name(0)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fp16", choices=["fp16", "nf4", "smoke"])
    ap.add_argument("--workloads", default="uniform,mixed")
    ap.add_argument("--systems", default="sequential,static,continuous_unbatched,continuous_batched")
    ap.add_argument("--caps", default="4,8")
    ap.add_argument("--rates", default="0.5,1,2",
                    help="open-loop rates as multiples of measured sequential capacity")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="n=8, repeats=2, caps=4, mixed only")
    ap.add_argument("--out", default="results/runs.jsonl")
    args = ap.parse_args()
    if args.quick:
        args.n, args.repeats, args.caps, args.workloads = 8, 2, "4", "mixed"
 
    systems = args.systems.split(",")
    caps = [int(c) for c in args.caps.split(",")]
    rates = [float(r) for r in args.rates.split(",")]
 
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok, model = load_model(args.model)
    print(f"device={model.device} torch={torch.__version__}", flush=True)
    warmup(model, tok)
 
    def configs():
        out = []
        for s in systems:
            for c in ([1] if s == "sequential" else caps):
                out.append((s, c))
        return out
 
    with open(args.out, "a") as f:
        f.write(json.dumps({"meta": meta(args)}) + "\n")
        for wl in args.workloads.split(","):
            for rep in range(args.repeats):
                seed = args.seed + rep
                workload = make_workload(wl, args.n, seed)
 
                def record(system, cap, arrival, res):
                    row = dict(workload=wl, repeat=rep, seed=seed, system=system,
                               cap=cap, arrival=arrival, **res)
                    f.write(json.dumps(row) + "\n"); f.flush()
                    print(f"[{wl} r{rep}] {system:22s} cap={cap} {arrival:>12s} "
                          f"{res['tok_per_s']:7.2f} tok/s  wall={res['wall']:.1f}s", flush=True)
 
                # sequential offline first: defines this stream's capacity mu
                res = run_once("sequential", model, tok, workload, 1, None, seed)
                mu = res["req_per_s"]
                record("sequential", 1, "offline", res)
 
                todo = [(s, c, "offline") for s, c in configs() if s != "sequential"]
                if not args.offline_only:
                    for r in rates:
                        todo += [(s, c, f"poisson@{r}") for s, c in configs()]
                random.Random(seed).shuffle(todo)
 
                for system, cap, arrival in todo:
                    offsets = None
                    if arrival.startswith("poisson@"):
                        rate = float(arrival.split("@")[1]) * mu
                        offsets = poisson_offsets(args.n, rate, seed)
                    res = run_once(system, model, tok, workload, cap, offsets, seed)
                    res["offered_rate"] = None if offsets is None else rate
                    res["seq_capacity_req_s"] = mu
                    record(system, cap, arrival, res)
 
 
if __name__ == "__main__":
    main()