"""
replot_pareto.py — recompute aggregates from an existing results.json and redraw
faithfulness_vs_cost.png. No model, no GPU, no re-eval.

The per-metric faithfulness scores are already stored; only the AGGREGATION changed
(infidelity is now in AGGREGATE_EXCLUDE, on the evidence in
INFIDELITY_FINDINGS_REPORT.md). So the figure can be rebuilt in seconds instead of
re-running a 3-hour eval.

Updates the "aggregate" field of every method in results.json and rewrites the PNG.
The original file is copied to results.json.pre_replot first.

    python replot_pareto.py --run A2_protopnet_mobilevit_120ep
    python replot_pareto.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

from pxai.eval.faithfulness import normalised_aggregates, AGGREGATE_EXCLUDE
from pxai.plots.pareto import plot_faithfulness_vs_cost


def replot(run_dir: str) -> bool:
    rj = os.path.join(run_dir, "results.json")
    if not os.path.exists(rj):
        print(f"  SKIP {run_dir} - no results.json")
        return False

    with open(rj) as f:
        results = json.load(f)
    if "methods" not in results or not results["methods"]:
        print(f"  SKIP {run_dir} - no methods recorded")
        return False

    shutil.copy(rj, rj + ".pre_replot")

    per_method = {k: v["faithfulness"] for k, v in results["methods"].items()}
    used = sorted({m for s in per_method.values() for m in s} - AGGREGATE_EXCLUDE)
    aggs = normalised_aggregates(per_method)

    rows = []
    for label, agg in aggs.items():
        old = results["methods"][label].get("aggregate")
        results["methods"][label]["aggregate"] = agg
        results["methods"][label]["aggregate_metrics"] = used
        rows.append({"method": label,
                     "passes": results["methods"][label]["passes"],
                     "faithfulness": agg,
                     "kind": "ante" if label.startswith("ours:") else "posthoc"})
        delta = "" if old is None else f"  (was {old:.3f})"
        print(f"    {label:>24} {agg:.3f}{delta}")

    with open(rj, "w") as f:
        json.dump(results, f, indent=2)
    png = os.path.join(run_dir, "faithfulness_vs_cost.png")
    plot_faithfulness_vs_cost(rows, png)
    print(f"  wrote {png}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run directory name under runs/")
    ap.add_argument("--all", action="store_true", help="every run with a results.json")
    args = ap.parse_args()

    if args.all:
        dirs = sorted(os.path.dirname(p) for p in glob.glob("runs/*/results.json"))
    elif args.run:
        dirs = [os.path.join("runs", args.run)]
    else:
        raise SystemExit("give --run <name> or --all")

    print(f"metrics EXCLUDED from the aggregate: {sorted(AGGREGATE_EXCLUDE)}\n")
    n = 0
    for d in dirs:
        print(f"{d}:")
        n += replot(d)
    print(f"\n{n} figure(s) regenerated")


if __name__ == "__main__":
    main()
