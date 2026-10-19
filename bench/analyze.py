"""
Turn results/*.jsonl into a summary CSV, a markdown table, and figures.

    python bench/analyze.py results/cpu.jsonl --out results/cpu_analysis

Statistics: per-run metrics are averaged over repeats (mean, 95% CI using a
Student-t critical value). Latency percentiles are pooled over all requests
from all repeats of a configuration. Output equivalence compares each
request's token sequence to the sequential engine's output for the same
(workload, seed, request id).
"""
import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict

T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
        8: 2.31, 9: 2.26, 10: 2.23}


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], None
    m = st.mean(xs)
    t = T975.get(len(xs) - 1, 1.96)
    return m, t * st.stdev(xs) / len(xs) ** 0.5


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load(path):
    runs, meta = [], None
    for line in open(path):
        d = json.loads(line)
        if "meta" in d:
            meta = d["meta"]
        else:
            runs.append(d)
    return meta, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta, runs = load(a.path)

    # reference outputs from the sequential engine
    ref = {}
    for r in runs:
        if r["system"] == "sequential" and r["arrival"] == "offline":
            for p in r["per_request"]:
                ref[(r["workload"], r["seed"], p["id"])] = p["out"]

    groups = defaultdict(list)
    for r in runs:
        groups[(r["workload"], r["arrival"], r["system"], r["cap"])].append(r)

    rows = []
    for (wl, arr, sysn, cap), rs in sorted(groups.items()):
        reqs = [p for r in rs for p in r["per_request"]]
        match = [p["out"] == ref[(r["workload"], r["seed"], p["id"])]
                 for r in rs for p in r["per_request"]
                 if (r["workload"], r["seed"], p["id"]) in ref]
        tps, tps_ci = mean_ci([r["tok_per_s"] for r in rs])
        rows.append(dict(
            workload=wl, arrival=arr, system=sysn, cap=cap, repeats=len(rs),
            tok_s=tps, tok_s_ci=tps_ci,
            req_s=mean_ci([r["req_per_s"] for r in rs])[0],
            ttft_p50=pct([p["ttft"] for p in reqs], .5),
            ttft_p99=pct([p["ttft"] for p in reqs], .99),
            e2e_p50=pct([p["e2e"] for p in reqs], .5),
            e2e_p99=pct([p["e2e"] for p in reqs], .99),
            tpot_p50=pct([p["tpot"] for p in reqs], .5),
            occupancy=mean_ci([r["mean_occupancy"] for r in rs])[0],
            rebuild_frac=mean_ci([r["rebuild_frac"] for r in rs])[0],
            rebuild_count=mean_ci([r["rebuild_count"] for r in rs])[0],
            wasted_slot_steps=mean_ci([r["wasted_slot_steps"] for r in rs])[0],
            output_match=(sum(match) / len(match)) if match else None,
        ))

    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def fmt(x, nd=2):
        return "-" if x is None else f"{x:.{nd}f}"

    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("| workload | arrival | system | cap | tok/s (±95% CI) | TTFT p50 | e2e p50 | e2e p99 "
                "| occupancy | rebuild+extract % | match vs seq |\n|" + "---|" * 11 + "\n")
        for r in rows:
            ci = "" if r["tok_s_ci"] is None else f" ± {r['tok_s_ci']:.2f}"
            rf = "-" if r["rebuild_frac"] is None else f"{100 * r['rebuild_frac']:.1f}"
            om = "-" if r["output_match"] is None else f"{100 * r['output_match']:.0f}%"
            f.write(f"| {r['workload']} | {r['arrival']} | {r['system']} | {r['cap']} | "
                    f"{fmt(r['tok_s'])}{ci} | {fmt(r['ttft_p50'])} | {fmt(r['e2e_p50'])} | "
                    f"{fmt(r['e2e_p99'])} | {fmt(r['occupancy'], 1)} | {rf} | {om} |\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: wrote tables only")
        return

    order = ["sequential", "static", "continuous_unbatched", "continuous_batched"]
    wls = sorted({r["workload"] for r in rows})

    # Fig 1: offline throughput by system and cap
    fig, axes = plt.subplots(1, len(wls), figsize=(5 * len(wls), 3.8), squeeze=False)
    for ax, wl in zip(axes[0], wls):
        sel = [r for r in rows if r["workload"] == wl and r["arrival"] == "offline"]
        labels = [(r["system"], r["cap"]) for s in order for r in sel if r["system"] == s]
        vals = [r for s in order for r in sel if r["system"] == s]
        ax.bar(range(len(vals)), [v["tok_s"] for v in vals],
               yerr=[v["tok_s_ci"] or 0 for v in vals], capsize=3)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"{s[:6]}\ncap{c}" for s, c in labels], fontsize=7)
        ax.set_title(f"{wl}: offline throughput")
        ax.set_ylabel("tokens/s")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_throughput.png"), dpi=160)

    # Fig 2: tail latency vs offered load (largest cap)
    fig, axes = plt.subplots(1, len(wls), figsize=(5 * len(wls), 3.8), squeeze=False)
    for ax, wl in zip(axes[0], wls):
        for s in order:
            pts = []
            for r in rows:
                if r["workload"] == wl and r["system"] == s and r["arrival"].startswith("poisson@"):
                    pts.append((float(r["arrival"].split("@")[1]), r["e2e_p99"], r["cap"]))
            if not pts:
                continue
            capmax = max(p[2] for p in pts)
            pts = sorted(p for p in pts if p[2] == capmax)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{s} (cap{capmax})")
        ax.set_xlabel("offered load (x sequential capacity)")
        ax.set_ylabel("p99 end-to-end latency (s)")
        ax.set_title(wl)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_latency.png"), dpi=160)

    # Fig 3: cache-management overhead share (continuous_batched, offline)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for wl in wls:
        pts = sorted((r["cap"], 100 * r["rebuild_frac"]) for r in rows
                     if r["workload"] == wl and r["system"] == "continuous_batched"
                     and r["arrival"] == "offline" and r["rebuild_frac"] is not None)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=wl)
    ax.set_xlabel("batch cap")
    ax.set_ylabel("% of wall time in cache rebuild/extract")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_rebuild.png"), dpi=160)
    print("wrote", a.out)


if __name__ == "__main__":
    main()