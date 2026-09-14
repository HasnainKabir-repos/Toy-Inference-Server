import time
from collections import deque
from dataclasses import dataclass, field
from transformers.cache_utils import DynamicCache
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_BATCH_SIZE = 4

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)
model.eval()

@dataclass
class Request:
    request_id: int
    prompt: str
    max_new_tokens: int

    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    next_token: torch.Tensor | None = None
    generated_tokens: list[int] = field(default_factory=list)

    past_key_values: object | None = None
    cache_length: int = 0

    done: bool = False
    finish_reason: str | None = None

    arrival_time: float | None = None
    batch_entry_time: float | None = None
    finish_time: float | None = None

    ttft: float | None = None
    decode_time: float = 0.0

    decode_steps: int = 0

    @property
    def waiting_time(self):
        if self.batch_entry_time is None:
            return None

        return self.batch_entry_time - self.arrival_time

    @property
    def total_time(self):
        if self.finish_time is None:
            return None

        return self.finish_time - self.arrival_time

    @property
    def generated_token_count(self):
        return len(self.generated_tokens)

    @property
    def text(self):
        return tokenizer.decode(
            self.generated_tokens,
            skip_special_tokens = True
        )

class RequestQueue:
    def __init__(self):
        self.queue = deque()

    def add(self, request):
        self.queue.append(request)

    def pop(self):
        if not self.queue:
            return None    
        return self.queue.popleft()

    def empty(self):
        return len(self.queue) == 0

    def __len__(self):
        return len(self.queue)


class ActiveBatch:
    def __init__(self, max_batch_size):
        self.max_batch_size = max_batch_size
        self.requests = []
        self.past_key_values = None
        self.cached_request_ids = None

    def add(self, request):
        if self.is_full:
            raise RuntimeError(
                "Cannot add request: active batch is full"
            )

        self.requests.append(request)

    def remove_finished(self):
        self.requests = [
            request
            for request in self.requests
            if not request.done
        ]

    @property
    def is_full(self):
        return len(self.requests) >= self.max_batch_size

    @property
    def empty(self):
        return len(self.requests) == 0

    def __len__(self):
        return len(self.requests)


class ContinuousBatchEngine:

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size=4,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.max_batch_size = max_batch_size
        self.waiting_queue = RequestQueue()

        self.active_batch = ActiveBatch(
            max_batch_size=max_batch_size
        )

        self.completed_requests = []
        self.cache_manager = KVCacheManager(model)

    def add_request(self, request):
        request.arrival_time = time.perf_counter()
        self.waiting_queue.add(request)

    def admit_requests(self):
        """ 
        Move requests from the waiting queue into the active batch.
        A newly admitted request performs PREFILL here. 
        """

        while(
            not self.active_batch.is_full and
            not self.waiting_queue.empty()
        ):
            request = self.waiting_queue.pop()
            self.prefill_request(request)
            request.batch_entry_time = time.perf_counter()
            self.active_batch.add(request)

    def prefill_request(self, request):
        """
        Process the entire prompt once. 
        This creates the initial KV cache and determines the first generated token.
        """

        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt"
        ).to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        request.input_ids = input_ids
        request.attention_mask = attention_mask
        request.cache_length = input_ids.shape[-1]

        if self.model.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True
            )

        if self.model.device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        request.ttft = end - start

        request.past_key_values = (
            outputs.past_key_values
        )
        logits = outputs.logits[:, -1, :]

        request.next_token = torch.argmax(
            logits,
            dim=-1
        )

    def decode_active_batch(self):
        """ 
        Perform ONE decode iteration for every currently active request. 
        This is the heart of continuous batching. 
        """

        requests = self.active_batch.requests
        if not requests:
            return

        cache_manager = KVCacheManager(self.model)
        current_request_ids = [r.request_id for r in requests]
        past_is_none = self.active_batch.past_key_values is None
        ids_differ = self.active_batch.cached_request_ids != current_request_ids
        needs_rebuild = past_is_none or ids_differ

        if needs_rebuild:
            (batched_cache,_) = cache_manager.build_batched_cache(requests)
            self.active_batch.past_key_values = batched_cache
            self.active_batch.cached_request_ids = current_request_ids.copy()
        else:
            batched_cache = self.active_batch.past_key_values

        cache_lengths = [
            request.cache_length
            for request in requests
        ]

        next_tokens = torch.stack(
            [
                request.next_token.squeeze()
                for request in requests
            ],
        ).unsqueeze(-1)

        attention_mask = (
            cache_manager.build_attention_mask(
                cache_lengths
            )
        )

        position_ids = (
            cache_manager.build_position_ids(
                cache_lengths
            )
        )

        if self.model.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(
                input_ids=next_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=batched_cache,
                use_cache=True,
            )

        if self.model.device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        batch_decode_time = (end - start)

        logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(
            logits,
            dim=-1
        )

        for i, request in enumerate(requests):
            generated_token = request.next_token
            request.generated_tokens.append(generated_token.item())
            request.decode_steps += 1
            request.cache_length += 1
            request.decode_time += batch_decode_time

            if (generated_token.item() == self.tokenizer.eos_token_id):
                request.done = True
                request.finish_reason = "eos"

            elif (request.decode_steps >= request.max_new_tokens):
                request.done = True
                request.finish_reason = "length"

            request.next_token = next_tokens[i]

        updated_cache = outputs.past_key_values
        self.active_batch.past_key_values = updated_cache
        self.active_batch.cached_request_ids = [r.request_id for r in requests]

        for i, request in enumerate(requests):
            request.past_key_values = cache_manager.extract_single_request_cache(
                updated_cache,
                i,
                request.cache_length
            )

    def decode_request(self, request:Request):
        """ 
        Decode exactly one token for one request. 
        This first implementation deliberately keeps each request's KV cache independent. 
        That makes the scheduler mechanics clear before optimizing the actual GPU batch operation. 
        """
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()    

        start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(
                input_ids=request.next_token.unsqueeze(-1),
                past_key_values=request.past_key_values,
                use_cache=True
            )

        if self.model.device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        request.decode_time += (end - start)
        request.decode_steps += 1

        token_id = request.next_token.item()
        request.generated_tokens.append(token_id)

        if token_id == self.tokenizer.eos_token_id:
            request.done = True
            request.finish_reason = "eos"
            return

        if (request.decode_steps >= request.max_new_tokens):
            request.done = True
            request.finish_reason = "length"
            return

        request.past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        request.next_token = torch.argmax(logits, dim=-1).squeeze(0)

    def finish_requests(self):
        """ Remove completed requests from the active batch. """
        old_requests = self.active_batch.requests

        if not old_requests:
            return

        remaining_requests = []
        remaining_indices = []

        for i, request in enumerate(old_requests):
            if request.done:
                request.finish_time = time.perf_counter()
                self.completed_requests.append(request)

            else:
                remaining_requests.append(request)
                remaining_indices.append(i)

        if self.active_batch.past_key_values is not None:
            if remaining_indices:
                indices = torch.tensor(
                    remaining_indices,
                    dtype=torch.long,
                    device=self.model.device
                )

                self.active_batch.past_key_values.batch_select_indices(indices)

            else:
                self.active_batch.past_key_values = None

        self.active_batch.requests = remaining_requests

    def run(self):
        """ Main continuous batching scheduler. """

        print() 
        print("=" * 70) 
        print("STARTING CONTINUOUS BATCHING") 
        print("=" * 70)

        while (
            not self.waiting_queue.empty()
            or not self.active_batch.empty
        ):
            self.admit_requests()
            self.decode_active_batch()
            self.finish_requests()

        print() 
        print("=" * 70) 
        print("CONTINUOUS BATCHING COMPLETE") 
        print("=" * 70)

        return self.completed_requests

class KVCacheManager:
    def __init__(self, model):
        self.model = model

    def get_cache_length(self, cache):
        """ 
        Return the number of cached sequence positions. 
        This assumes the legacy tuple-style cache: 
            ( 
                (key_layer_0, value_layer_0), 
                (key_layer_1, value_layer_1), 
                ... 
            ) 
        We will verify/adapt this if your installed 
        Transformers version returns a Cache object. 
        """
        return cache.get_seq_length()

    def build_batched_cache(self, requests:list[Request]):
        """
        Combine individual DynamicCaches into one DynamicCache.

        Example:

            R1: length 5
            R2: length 7
            R3: length 10
            R4: length 9

        becomes a batch with cache length 10.

        Shorter caches are LEFT padded so that the real
        cached tokens are aligned at the right edge.
        """
        caches = [
            request.past_key_values
            for request in requests
        ]

        cache_lengths = [
            cache.get_seq_length()
            for cache in caches
        ]
        max_cache_length = max(cache_lengths)

        batched_cache = DynamicCache()

        for layer_idx in range(len(caches[0].layers)):
            keys = []
            values = []

            for cache, cache_length in zip(caches, cache_lengths):
                layer = cache.layers[layer_idx]
                key = layer.keys
                value = layer.values

                padding_length = (max_cache_length - cache_length)

                if padding_length > 0:
                    key_padding = torch.zeros(
                        (
                            key.shape[0],
                            key.shape[1],
                            padding_length,
                            key.shape[3]
                        ),
                        dtype=key.dtype,
                        device=key.device
                    )

                    value_padding = torch.zeros(
                        (
                            value.shape[0],
                            value.shape[1],
                            padding_length,
                            value.shape[3]
                        ),
                        dtype=value.dtype,
                        device=value.device
                    )

                    key = torch.cat(
                        [
                            key_padding,
                            key
                        ],
                        dim=-2
                    )

                    value = torch.cat(
                        [
                            value_padding,
                            value
                        ],
                        dim=-2
                    )

                keys.append(key)
                values.append(value)

            batch_keys = torch.cat(keys, dim=0)
            batch_values = torch.cat(values, dim=0)

            batched_cache.update(batch_keys, batch_values, layer_idx)

        return batched_cache, cache_lengths

    def build_attention_mask(self, cache_lengths):
        """
        Attention mask covers:

            [cached positions] + [current token]

        Example:

            cache lengths = [5, 7, 10, 9]
            max length = 10

            R1:
            0 0 0 0 0 1 1 1 1 1 1

            R2:
            0 0 0 1 1 1 1 1 1 1 1

            R3:
            1 1 1 1 1 1 1 1 1 1 1

            R4:
            0 1 1 1 1 1 1 1 1 1 1

        """

        max_cache_length = max(cache_lengths)
        masks = []
        for length in cache_lengths:
            padding = (max_cache_length - length)
            mask = torch.cat(
                [
                    torch.zeros(
                        padding,
                        dtype=torch.long
                    ),
                    torch.ones(
                        length + 1,
                        dtype=torch.long
                    )
                ]
            )
            masks.append(mask)

        return torch.stack(masks).to(self.model.device)

    def build_position_ids(self, cache_lengths):
        """ The newly decoded token's position is: 
            current cache length 
        Example: 
            R1 cache length = 20 
            R2 cache length = 30 
            position_ids: [20] [30] 
        """

        return torch.tensor(
            cache_lengths,
            dtype=torch.long,
            device=self.model.device
        ).unsqueeze(-1)

    def extract_request_cache(self, batched_cache, batch_index):
        """Extract one requst from the batched cache"""
        legacy_cache = self.to_legacy_cache(batched_cache)
        request_cache = []

        for key, value in legacy_cache:
            key = key[batch_index: batch_index + 1]
            value = value[batch_index: batch_index + 1]
            request_cache.append((key, value))
        return tuple(request_cache)

    def extract_single_request_cache(
        self,
        batched_cache,
        index,
        cache_length,
    ):
        """Extract only the real KV tokens belonging to one request."""

        single_cache = DynamicCache()
        physical_length = batched_cache.get_seq_length()
        padding_length = physical_length - cache_length

        for layer_idx in range(len(batched_cache.layers)):
            layer = batched_cache.layers[layer_idx]

            key = layer.keys[
                index:index + 1,
                :,
                padding_length:,
                :
            ].clone()

            value = layer.values[
                index:index + 1,
                :,
                padding_length:,
                :
            ].clone()

            single_cache.update(
                key,
                value,
                layer_idx,
            )

        return single_cache
        
def print_results(results: list[Request]):
    print() 
    print("=" * 70) 
    print("RESULTS") 
    print("=" * 70)

    for request in sorted(results, key=lambda r: r.request_id):
        print()
        print(f"Request {request.request_id}")
        print(f"Prompt: {request.prompt}")
        print(f"Output: {request.text!r}")
        print(f"Generated tokens: {request.generated_token_count}")
        print(f"Finish reason: {request.finish_reason}")
        print(f"Waiting time: {request.waiting_time:.4f}s")
        print(f"TTFT: {request.ttft:.4f}")
        print(f"Decode time: {request.decode_time:.4f}s")
        print(f"Total time: {request.total_time:.4f}s")


if __name__ == "__main__":
    prompts = [ 
        "The capital of France is", 
        "Explain what a CPU does.", 
        "The history of artificial intelligence can be traced back to", 
        "What is the difference between RAM and storage?", 
        "In machine learning, gradient descent is used to", 
        "Explain how a transformer model works in detail.", 
        "The capital of Japan is", 
        "Why is the sky blue?", 
    ]

    max_new_tokens = [ 15, 50, 20, 50, 30, 50, 10, 50 ]

    engine = ContinuousBatchEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=MAX_BATCH_SIZE,
    )

    for i, prompt in enumerate(prompts):
        request = Request(
            request_id=i+1,
            prompt=prompt,
            max_new_tokens=max_new_tokens[i]
        )

        engine.add_request(request)

    benchmark_start = time.perf_counter()
    results = engine.run()
    benchmark_end = time.perf_counter()

    wall_time = (
        benchmark_end - benchmark_start
    )

    print_results(results)
    print()
    print("=" * 70)
    print("SYSTEM METRIC")
    print("=" * 70)

    total_tokens = sum(
        request.generated_token_count
        for request in results
    )

    total_decode_time = sum(
        request.decode_time
        for request in results
    )

    throughput = (
        total_tokens / wall_time
        if wall_time > 0
        else 0
    )

    print(f"Requests: {len(results)}")
    print(f"Max batch size: {MAX_BATCH_SIZE}")
    print(f"Wall-clock time: {wall_time:.4f}")
    print(f"Sum of per-request decode time: {total_decode_time}")
    print(f"System throughput: {throughput:.2f} token/s")