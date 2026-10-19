# bench/ — controlled study harness

    pip install matplotlib            # for analyze.py figures

    # 1. sanity check, no model download (tiny random model)
    python bench/bench.py --model smoke --quick --out results/smoke.jsonl

    # 2. core study, CPU (3-5 h; start with --repeats 3 if short on time)
    python bench/bench.py --model fp16 --workloads uniform,mixed --caps 4,8 \
        --repeats 5 --n 24 --out results/cpu.jsonl

    # 3. full matrix on the T4
    python bench/bench.py --model fp16 --workloads uniform,mixed,longctx \
        --caps 2,4,8 --repeats 5 --n 32 --out results/t4.jsonl

    # 4. tables + figures
    python bench/analyze.py results/cpu.jsonl --out results/cpu_analysis

Systems: sequential, static, continuous_unbatched, continuous_batched.
Modes: offline (all at t=0) and open-loop Poisson at 0.5x/1x/2x of the
measured sequential capacity. Results are one JSON line per run.