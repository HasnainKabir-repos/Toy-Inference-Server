# results/

Raw benchmark data (one JSON line per run; the first line of each file is metadata) and the analysis derived from it.

## raw/

| File | Systems | Seeds | What it is |
|---|---|---|---|
| `cpu_mixed.jsonl` | sequential, static, continuous_unbatched | 0-2 | Session 1, mixed output lengths |
| `cpu_uniform.jsonl` | sequential, static, continuous_unbatched | 0-2 | Session 1, uniform output lengths |
| `final_mixed_cb.jsonl` | sequential, continuous_batched | 0-4 | Session 2, mixed lengths |
| `final_uniform_cb.jsonl` | sequential, continuous_batched | 0-4 | Session 2, uniform lengths |
| `smoke_after.jsonl` | sequential, static, continuous_batched | 0-2 | Tiny random float32 model; **correctness check only** |

All real runs: Qwen3-0.6B, float16, CPU, batch cap 4, 24 requests per run, open-loop rates 1x/2x/3x of the sequential capacity.
Rows from an earlier revision of the batched engine were removed from the `cpu_*` files; the batched engine's results
come from the `final_*_cb` files.

`smoke_after.jsonl` has throughput numbers that mean nothing (the model is tiny and random). Use it only for the
output-agreement column.

## Commands that produced the data

```bash
# session 1 (cpu_*)
python bench/bench.py --model fp16 --workloads mixed   --caps 4 --rates 1,2,3 --n 24 --repeats 3 --out results/raw/cpu_mixed.jsonl
python bench/bench.py --model fp16 --workloads uniform --caps 4 --rates 1,2,3 --n 24 --repeats 3 --out results/raw/cpu_uniform.jsonl

# session 2 (final_*_cb)
python bench/bench.py --model fp16 --workloads mixed   --caps 4 --rates 1,2,3 --n 24 --repeats 5 --systems sequential,continuous_batched --out results/raw/final_mixed_cb.jsonl
python bench/bench.py --model fp16 --workloads uniform --caps 4 --rates 1,2,3 --n 24 --repeats 5 --systems sequential,continuous_batched --out results/raw/final_uniform_cb.jsonl

# correctness check on the tiny float32 model
python bench/bench.py --model smoke --workloads uniform,mixed --caps 4 --rates 1,2,3 --n 24 --repeats 3 --systems sequential,static,continuous_batched --out results/raw/smoke_after.jsonl
```

(The `cpu_*` commands were run on the earlier revision of the batched engine, which is why its rows were removed.)

## Regenerating the analysis

```bash
# four-system comparison on seeds 0-2 (cpu_* for sequential/static/unbatched, final_*_cb for the batched engine)
python bench/merge.py   --results results/raw --out results/combined_3seed.jsonl
python bench/analyze.py results/combined_3seed.jsonl --out results/analysis/combined

# sequential vs batched on seeds 0-4
python bench/analyze.py results/raw/final_mixed_cb.jsonl   --out results/analysis/final_mixed_cb
python bench/analyze.py results/raw/final_uniform_cb.jsonl --out results/analysis/final_uniform_cb

# which requests diverge from the sequential output, and where
python bench/diff.py results/raw/final_uniform_cb.jsonl
```

`combined_3seed.jsonl` is generated and not committed.

## Which paper table comes from where

| Paper | Source |
|---|---|
| Table 1 (offline throughput, four systems) and Table 3 (latency under load) | `analysis/combined` |
| Table 2 (sequential vs batched, 5 repeats) and the cache-overhead numbers in section 5.4 | `analysis/final_*_cb` |
| Section 6 (correctness) | `raw/smoke_after.jsonl` and the output of `bench/diff.py` |

## Reading the tables

- **tok/s** is realized tokens divided by makespan; the interval is a 95% CI over repeats (Student-t).
- **At 1x offered load**, tok/s mostly reflects the arrival rate, so read latency instead.
- **Latency percentiles** are pooled over all requests from all repeats, so they have no interval and p99 rests on tens of requests.
- **match vs seq** is the share of requests whose token sequence equals the sequential engine's for the same request. In float16, agreement below 100% for the batched engine comes from a few near-tied prompts, not from batching logic (float32 agrees 100%).
- **occupancy** is the mean number of live requests per decode step. **rebuild+extract %** is wall time spent building and extracting batched caches.
- Open-loop rates are multiples of the sequential capacity measured in each run, so offered load differs slightly between sessions.