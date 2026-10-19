"""
Generate the three figures referenced in the paper.

    python bench/paper_figures.py \
        --combined results/combined_3seed.jsonl \
        --final-mixed results/final_mixed_cb.jsonl \
        --final-uniform results/final_uniform_cb.jsonl \
        --out results/paper_figs

Outputs: fig1_offline_throughput.png, fig2_latency_vs_load.png,
         fig3_cache_overhead_occupancy.png
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYSTEMS = ["sequential", "continuous_unbatched", "static", "continuous_batched"]
LABEL = {"sequential": "Sequential", "continuous_unbatched": "Continuous, unbatched",
         "static": "Static", "continuous_batched": "Continuous, batched"}
COLOR = {"sequential": "#7f7f7f", "continuous_unbatched": "#bcbd22",
         "static": "#ff7f0e", "continuous_batched": "#1f77b4"}
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36}
WORKLOADS = ["mixed", "uniform"]


def load(path):
    return [d for d in map(json.loads, open(path)) if "system" in d]


def mean_ci(xs):
    m = st.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, T975.get(len(xs) - 1, 1.96) * st.stdev(xs) / len(xs) ** 0.5


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", required=True)
    ap.add_argument("--final-mixed", required=True)
    ap.add_argument("--final-uniform", required=True)
    ap.add_argument("--out", default="results/paper_figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    comb = load(a.combined)
    fin = load(a.final_mixed) + load(a.final_uniform)

    # ---- Figure 1: offline throughput, 4 systems x 2 workloads
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    top = 0
    for ax, wl in zip(axes, WORKLOADS):
        for i, s in enumerate(SYSTEMS):
            v = [r["tok_per_s"] for r in comb
                 if r["workload"] == wl and r["system"] == s and r["arrival"] == "offline"]
            m, ci = mean_ci(v)
            ax.bar(i, m, yerr=ci, capsize=4, color=COLOR[s])
            ax.text(i, m + ci + 0.4, f"{m:.1f}", ha="center", fontsize=8)
            top = max(top, m + ci)
        ax.set_xticks(range(len(SYSTEMS)))
        ax.set_xticklabels([LABEL[s].replace(", ", ",\n").replace(" ", "\n", 1) if s == "sequential" else LABEL[s].replace(", ", ",\n") for s in SYSTEMS], fontsize=7)
        ax.set_title(f"{wl.capitalize()} output lengths")
    axes[0].set_ylim(0, top * 1.12)
    axes[0].set_ylabel("Throughput (tokens/s)")
    fig.suptitle("Offline throughput, batch cap 4 (mean and 95% CI, 3 repeats)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig1_offline_throughput.png"), dpi=200)

    # ---- Figure 2: latency vs offered load
    loads = [1.0, 2.0, 3.0]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for col, wl in enumerate(WORKLOADS):
        for row, (key, title) in enumerate([("ttft", "Median time to first token (s)"),
                                            ("e2e", "Median end-to-end latency (s)")]):
            ax = axes[row][col]
            for s in SYSTEMS:
                ys = []
                for L in loads:
                    vals = [p[key] for r in comb if r["workload"] == wl and r["system"] == s
                            and r["arrival"] == f"poisson@{L}" for p in r["per_request"]]
                    ys.append(pct(vals, .5))
                ax.plot(loads, ys, marker="o", color=COLOR[s], label=LABEL[s])
            ax.set_ylabel(title if col == 0 else "")
            if row == 0:
                ax.set_title(f"{wl.capitalize()} output lengths")
                ax.set_yscale("log")
            if row == 1:
                ax.set_xlabel("Offered load (multiple of sequential capacity)")
            ax.set_xticks(loads)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Latency under open-loop Poisson arrivals, batch cap 4 (pooled over 3 repeats)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig2_latency_vs_load.png"), dpi=200)

    # ---- Figure 3: (a) cache overhead vs load, (b) slot occupancy offline
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    arrivals = ["offline", "poisson@1.0", "poisson@2.0", "poisson@3.0"]
    xl = ["Offline", "1x", "2x", "3x"]
    for wl, mk in zip(WORKLOADS, ["o", "s"]):
        pts = []
        for arr in arrivals:
            v = [100 * r["rebuild_frac"] for r in fin if r["workload"] == wl
                 and r["system"] == "continuous_batched" and r["arrival"] == arr]
            pts.append(mean_ci(v))
        ax1.errorbar(range(4), [p[0] for p in pts], yerr=[p[1] for p in pts],
                     marker=mk, capsize=3, label=wl.capitalize())
    ax1.set_xticks(range(4)); ax1.set_xticklabels(xl)
    ax1.set_ylabel("Wall time spent on cache build/extract (%)")
    ax1.set_xlabel("Arrival mode / offered load")
    ax1.set_title("(a) Cache-management overhead", fontsize=10)
    ax1.legend(fontsize=8)
    w = 0.35
    for j, s in enumerate(["static", "continuous_batched"]):
        vals = []
        for wl in WORKLOADS:
            v = [r["mean_occupancy"] for r in comb if r["workload"] == wl
                 and r["system"] == s and r["arrival"] == "offline"]
            vals.append(st.mean(v))
        bars = ax2.bar([i + (j - .5) * w for i in range(2)], vals, w,
                       color=COLOR[s], label=LABEL[s])
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + .05, f"{v:.1f}", ha="center", fontsize=8)
    ax2.axhline(4, ls="--", color="k", lw=.8)
    ax2.set_xticks(range(2)); ax2.set_xticklabels([w_.capitalize() for w_ in WORKLOADS])
    ax2.set_ylim(0, 4.6)
    ax2.set_ylabel("Mean occupied slots (of 4)")
    ax2.set_title("(b) Slot occupancy, offline", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig3_cache_overhead_occupancy.png"), dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()