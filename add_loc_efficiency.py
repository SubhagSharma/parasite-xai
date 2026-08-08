"""add_loc_efficiency.py — rescore figs/attribution_metrics.tsv. No GPU, no rerun.

    python add_loc_efficiency.py
    python add_loc_efficiency.py --tsv figs/attribution_metrics.tsv --write

WHY. `conc` is bounded above by 1 / box-area, so its ceiling is a property of
the DATASET, not of the explainer:

    chula_crops      mean box area 0.546  ->  conc can never exceed  1.9
    chula_roi2_w477                 0.118                           16.2
    chula_roi2_w679                 0.057                           33.8
    data (whole)                    0.024                          151.5

A conc of 1.34 on crops and a conc of 11.3 on whole images are not 1.34 and
11.3 -- they are 70% and 7% of their respective ceilings. Any table that puts
raw conc from two datasets in the same column is comparing different scales,
and any cross-dataset ordering read off it is an artefact.

THE FIX. Chance-correct it the way kappa does, (observed - chance) / (best -
chance). With frac = share of attribution mass inside the box and a = box area:

    eps = (frac - a) / (1 - a)              [ = (conc - 1) / (1/a - 1) ]

    eps = 0   attribution is uniform (mass in box exactly proportional to area)
    eps = 1   all mass inside the box
    eps < 0   the explainer puts LESS mass on the egg than a blank map would

eps is comparable across datasets, crop settings and box sizes. Report it
alongside conc; do not silently replace conc, since the report already quotes
conc figures.

Where only conc_pos is stored, frac_pos = conc_pos * a recovers frac.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def add_eps(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    a = d["area"].astype(float)
    if "frac" in d:
        d["eps"] = (d["frac"] - a) / (1 - a)
    if "conc_pos" in d:
        d["eps_pos"] = (d["conc_pos"] * a - a) / (1 - a)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="figs/attribution_metrics.tsv")
    ap.add_argument("--write", action="store_true",
                    help="write <tsv>.eps.tsv with the new columns")
    ap.add_argument("--min-n", type=int, default=30)
    a = ap.parse_args()

    d = pd.read_csv(a.tsv, sep="\t")
    d = add_eps(d)
    ok = d[np.isfinite(d.get("eps_pos", d["eps"]))]
    col = "eps_pos" if "eps_pos" in ok else "eps"

    print("ceiling of conc per dataset (= 1 / mean box area)\n")
    print(ok.groupby("dataset").agg(n=(col, "size"), box_area=("area", "mean"),
                                    conc_ceiling=("area", lambda s: (1 / s).mean())
                                    ).round(3).to_string())

    print("\n\nlocalisation efficiency by dataset x method "
          "(0 = uniform, 1 = all mass on the egg)\n")
    for ds, g in ok.groupby("dataset"):
        t = g.groupby("method").agg(
            conc_pos=("conc_pos", "mean"), eps=(col, "mean"),
            ci=(col, lambda s: 1.96 * s.std(ddof=1) / np.sqrt(len(s))),
            peak=("peak", "mean"), n=(col, "size"))
        t = t[t.n >= a.min_n].sort_values("eps", ascending=False)
        print(f"--- {ds}   (mean box area {g.area.mean():.3f})")
        print(t.round(3).to_string(), "\n")

    print("\nPAIRED within-run comparison -- the only fair one, since the "
          "ante-hoc row\nexists in some runs and not others\n")
    for ds, g in ok.groupby("dataset"):
        runs = g[g.method.str.startswith("ours:")].run.unique()
        if not len(runs):
            continue
        t = g[g.run.isin(runs)].pivot_table(index="run", columns="method",
                                            values=col, aggfunc="mean")
        print(f"--- {ds}")
        print(t.round(3).to_string(), "\n")

    if a.write:
        out = a.tsv.replace(".tsv", ".eps.tsv")
        d.to_csv(out, sep="\t", index=False)
        print(f"wrote {out}  ({len(d)} rows)")


if __name__ == "__main__":
    main()
