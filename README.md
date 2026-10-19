# Toy Inference Server

A small from-scratch LLM inference server (Qwen3-0.6B, PyTorch + Transformers) built to study
**what batching and KV-cache management actually do to throughput and latency**. It implements four
serving strategies and a benchmark harness that compares them under controlled conditions.

The write-up is in [`paper/`](paper/). This README is the short version.

## The four strategies

| System | What it does |
|---|---|
| `sequential` | One request at a time, FIFO, per-request KV cache |
| `static` | Pad up to B requests, run until the longest finishes, release all together |
| `continuous_unbatched` | Admit/retire requests every step, but one forward pass per request (scheduling only) |
| `continuous_batched` | Admit/retire every step and decode all active requests in one forward pass over a merged, left-padded KV cache |

`run.py` contains the continuous-batching engine and KV cache manager. The other three strategies and the
instrumentation live in `bench/engines.py`.

## Headline results

CPU, float16, batch cap B = 4, 24 requests per run. Full tables, methodology and limitations are in the paper.

- **Scheduling alone does not raise throughput.** `continuous_unbatched` matches `sequential` (about 9.2 tok/s) in every configuration.
- **Batched decode roughly doubles throughput.** 1.96x (mixed output lengths) and 2.36x (uniform lengths) offline over sequential, 5 repeats.
- **Continuous vs static depends on length variety.** On uniform lengths static is as fast or slightly faster. On mixed lengths continuous batching has about 3x lower median end-to-end latency at 2x the sequential capacity.
- **Cache rebuild/extract costs 5 to 7% of wall time** at short (about 40-token) contexts. Long contexts were not tested.

Caveats: one small model, one CPU, short prompts, 3 to 5 repeats. Several throughput intervals overlap (notably static vs continuous on mixed lengths). See the paper's limitations section.

## Quick start

```bash
pip install -r requirements.txt

# sanity check with a tiny random model (no download, takes seconds)
python bench/bench.py --model smoke --quick --out results/smoke_check.jsonl

# one real experiment (several hours on CPU; see results/README.md for the exact commands behind the paper)
python bench/bench.py --model fp16 --workloads mixed --caps 4 --rates 1,2,3 --n 24 --repeats 3 --out results/my_run.jsonl

# tables and figures
python bench/analyze.py results/my_run.jsonl --out results/analysis/my_run
```

## How the benchmark works

- Every system sees the **identical, seeded request stream** in each repeat.
- **Offline** mode: all requests at t = 0. **Open-loop** mode: Poisson arrivals at 1x/2x/3x the sequential engine's measured capacity, with latency measured from the *scheduled* arrival time (no coordinated omission).
- Warmup, shuffled run order, repeats, greedy decoding, one JSON line per run (with metadata: versions, hardware, commit).
- **Output-equivalence check:** every request's tokens are compared with the sequential engine's output. On a float32 tiny model the batched engines agree 100%. In float16 the batched engine agrees on 92 to 100% of requests; the divergences are at tokens 15/16 of a few specific prompts (see the paper, section 6).

## Repository layout

```
run.py              continuous-batching engine and KV cache manager
bench/              benchmark harness (engines, workloads, runner, analysis, merge, diff)
results/            raw per-run data and analysis; see results/README.md
paper/              the write-up
legacy/             earlier HTTP server and metrics poller (not maintained)
```

## Known limitations of the code

- The KV cache is a contiguous, left-padded tensor that is rebuilt when batch membership changes. This is simple and measurable, but it is the cost that block-based (paged) caches avoid.
- Per-request caches are cloned out of the batch cache whenever requests are waiting, which is wasteful when the batch is full.
- Greedy decoding only; no sampling, no streaming API.