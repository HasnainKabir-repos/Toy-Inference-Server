# Toy Continuous-Batching Inference Server

A from-scratch, educational reimplementation of the core scheduling and batching mechanics behind modern LLM inference servers (vLLM, TGI, etc.) — built to understand *why* these systems work the way they do, by implementing the pieces rather than reading someone else's production code.

Built with plain Python and HuggingFace `transformers` on CPU, no vLLM code involved. Model: `Qwen/Qwen3-0.6B`.

## What's implemented

The project is built as a sequence of stages, each building on the last:

| Stage | What it adds | Result |
|---|---|---|
| 1. Single request | Hand-written token-by-token generation loop (no `model.generate()`) | Baseline correctness |
| 2. Sequential queue | Multiple requests processed one at a time, with KV cache and TTFT/decode/TPS instrumentation | 43.2s / 8 requests, ~9.7 tok/s |
| 3a. Static batching | All requests padded and batched together for the full generation | ~30 tok/s (~3.4x speedup) |
| 3b. Continuous batching (scheduling only) | Dynamic admission/eviction as requests finish, but decode still runs as separate unbatched passes | Confirms scheduling alone doesn't improve throughput without batched compute |
| 4. Continuous batching (real batched decode) | Decode across the active batch merged into a single padded forward pass per step, with a `KVCacheManager` handling variable-length caches across requests that joined the batch at different times | ~16 tok/s system throughput |

## Architecture

- **`Request`** — per-request state: prompt, KV cache, generated tokens, timing (arrival, batch entry, finish), finish reason.
- **`RequestQueue`** — simple FIFO waiting queue.
- **`ActiveBatch`** — the currently-in-flight set of requests being decoded together.
- **`ContinuousBatchEngine`** — the scheduler: admits requests into free slots, runs one decode step across the active batch, evicts finished requests, repeats until the queue and active batch are both empty.
- **`KVCacheManager`** — builds a shared left-padded `DynamicCache` from requests with different individual cache lengths, and computes the per-row `attention_mask`/`position_ids` needed for a correct batched forward pass each step.

## Key things this project surfaces (not just implements)

- **KV caching matters for more than speed** — the naive Stage 1 loop recomputes the full sequence every step (O(n²)); `past_key_values` avoids that.
- **Padding requires care.** Left-padding, attention masks, and *manually computed* `position_ids` (not the model's implicit defaults) are all required for correct batched generation — getting any one wrong produces plausible-looking but subtly or badly wrong output.
- **Batched vs. sequential inference isn't bit-exact**, even when implemented correctly — floating-point non-associativity in batched attention kernels can flip close argmax ties, causing generations to diverge after matching for a while. This is a known open problem in LLM serving, not a bug to "fix."
- **Scheduling and batching are separable, and the difference is measurable.** Stage 3b (smarter admission order, no batched compute) shows no throughput gain over sequential — proving that *fairness* and *throughput* are genuinely different problems, and that continuous batching's real value comes from merging compute, not just reordering it.
- **Merging variable-length KV caches is the hard part.** Requests joining a batch at different times have different cache lengths; naively deriving "how much padding does this row have" from the shared tensor's shape (rather than tracking each request's real content length explicitly) silently corrupts attention — a bug this project hit, diagnosed, and fixed.
- **Why production systems use block-based KV cache management (PagedAttention).** This project's pad-and-copy approach rebuilds the entire batched cache from scratch on every admission/eviction — functional, but the exact inefficiency that motivates allocating fixed-size cache blocks instead.

## Known limitations / open items

- Occasional missing-space artifacts in generated text (e.g. "Whatis"), suspected but not confirmed to correlate with decode steps immediately following a batch re-composition event.
- No block-based KV cache management — the current approach re-pads and re-copies the full batch cache on every membership change.
- No load-testing/benchmarking harness comparing this server's behavior against real vLLM under equivalent conditions.
- CPU-only; GPU-specific effects (e.g. true parallel batched matmul speedup) aren't represented in the throughput numbers here.

## Running it

```bash
pip install transformers torch
python continuous_batch_engine.py
```

Edit the `prompts` and `max_new_tokens` lists in `__main__` to change the workload. `MAX_BATCH_SIZE` controls how many requests can be active at once.

## Motivation / next steps

The plan going forward is to apply what this project taught about reading and reasoning through scheduler/batching code to contributing to a smaller, more readable OSS project (e.g. `guidellm`) rather than vLLM's core engine directly, and to keep building — block-based cache management being the natural next piece to implement.