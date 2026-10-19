"""
Build the 3-seed, four-system dataset from two sessions.

  sequential / static / continuous_unbatched : results/cpu_<wl>.jsonl  (seeds 0-2)
  continuous_batched (patched engine)        : results/final_<wl>_cb.jsonl (seeds 0-2)

Rows of the OLD continuous_batched (pre-fix) in cpu_<wl>.jsonl are dropped.
Each row gets a "source" field recording which file it came from.

    python bench/merge.py --results results --out results/combined_3seed.jsonl
"""
import argparse
import json
import os

SEEDS = {0, 1, 2}

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/combined_3seed.jsonl")
ap.add_argument("--workloads", default="mixed,uniform")
a = ap.parse_args()


def rows(path):
    out = []
    for line in open(path):
        d = json.loads(line)
        if "system" in d:
            out.append(d)
    return out


n = {}
with open(a.out, "w") as f:
    f.write(json.dumps({"meta": {"note": "merged from two sessions; see merge.py"}}) + "\n")
    for wl in a.workloads.split(","):
        base_path = os.path.join(a.results, f"cpu_{wl}.jsonl")
        cb_path = os.path.join(a.results, f"final_{wl}_cb.jsonl")
        for r in rows(base_path):
            if r["seed"] in SEEDS and r["system"] != "continuous_batched":
                r["source"] = os.path.basename(base_path)
                f.write(json.dumps(r) + "\n")
                n[(wl, r["system"])] = n.get((wl, r["system"]), 0) + 1
        for r in rows(cb_path):
            if r["seed"] in SEEDS and r["system"] == "continuous_batched":
                r["source"] = os.path.basename(cb_path)
                f.write(json.dumps(r) + "\n")
                n[(wl, r["system"])] = n.get((wl, r["system"]), 0) + 1

for k, v in sorted(n.items()):
    print(k, v, "rows")