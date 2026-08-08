"""
probe_topk_localisation.py — does the explanation RANK the egg highly, even if it does
not put its MASS there?

THE TENSION THIS RESOLVES  (technical report §7.4)
--------------------------------------------------
Three measurements of roi477_protopnet cannot all be right:

    displaced control      94.9% egg-specific  -> the model NEEDS the egg
    deletion               0.0038, best of 22  -> the explanation is HIGHLY FAITHFUL
    conc_pos               1.04, peak 24%      -> the explanation is NOT ON the egg

The most likely resolution is that these measure different things:

    DELETION removes pixels in ATTRIBUTION ORDER. It only cares about the ranking near
    the top -- the first few percent of pixels removed. It is indifferent to where the
    remaining 95% of attribution mass sits.

    conc_pos is a MASS statistic. A map with a high floor spreads mass frame-wide and
    scores ~1.0 no matter how well it ranks.

These are not contradictory. A map whose TOP pixels are on the egg, but whose bulk mass
is smeared everywhere, will delete superbly and concentrate at uniform. That is a
coherent object, and if it is what ProtoPNet produces then §7.4 dissolves and the
correct statement changes from "the explanation does not find the egg" to "the
explanation finds the egg but cannot draw a tight boundary around it" -- a much weaker
and more defensible criticism.

WHAT IT MEASURES
----------------
For each attribution map, take the top k% of pixels by attribution value and ask what
fraction fall inside the annotation box.

    topk_frac(k)   fraction of the top k% of pixels that are inside the box
    topk_conc(k)   topk_frac(k) / area   -- 1.0 = no better than uniform
    best_rank_pct  percentile rank of the highest-ranked pixel that IS inside the box.
                   0.0 means the single highest-attribution pixel in the image is on the
                   egg. 50.0 means you must descend halfway through the ranking before
                   the explanation points at the egg even once.

`best_rank_pct` is the sharpest of the three: it is floor-free, scale-free, and answers
"does this explanation ever point at the egg first?" with one number.

READING THE RESULT
------------------
    topk_conc(1%) >> conc_pos          -> the tension DISSOLVES. The explanation ranks
                                          the egg highly and spreads its mass. Deletion
                                          and conc_pos were measuring different things
                                          and both are correct.
    topk_conc(1%) ~= conc_pos ~= 1.0   -> the tension is REAL. The explanation neither
                                          ranks nor locates the egg, and the deletion
                                          score needs auditing (§7.4 resolution 2).

USAGE
    # the four ROI heads, ante-hoc + the best post-hoc comparators   (~12 min on GPU)
    python -u probe_topk_localisation.py \
        --runs 'roi477_protopnet_120ep,roi477_bcos_120ep,roi477_cbm_sup_120ep,roi477_blackbox_120ep' \
        --emit-tsv figs/topk_localisation.tsv

    # add the slow sampling methods (~50 min)
    python -u probe_topk_localisation.py --runs '...' --methods gradcam,hirescam,integrated_gradients,lime,kernelshap
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics as st

import numpy as np
import torch
from torch.utils.data import Subset

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.evaluate import ante_hoc_attr
from pxai.explainers.posthoc import explain_posthoc
from pxai.eval.cropgeom import load_coco, box_in_crop

KS = (0.001, 0.01, 0.05, 0.10)          # top 0.1%, 1%, 5%, 10%


def labels_for(root):
    """Dataset-local labels.json if present; the chula_roi2_* sets carry remapped boxes."""
    local = os.path.join(root, "labels.json")
    if os.path.exists(local):
        return local
    up = os.path.join(os.path.dirname(root.rstrip("/")), "Chula-ParasiteEgg-11",
                      "labels.json")
    return up if os.path.exists(up) else None


def topk_metrics(attr, mask):
    """-> area, {k: frac}, best_rank_pct, mass_conc.

    Ranking uses the SIGNED attribution descending, i.e. strongest positive evidence
    first, which is the order deletion removes pixels in. Using |a| would put strong
    negative evidence at the top and measure a different thing.
    """
    if mask is None:
        return None
    a = np.asarray(attr, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    n = a.size
    if not np.isfinite(a).all() or n == 0:
        return None
    area = float(m.mean())
    order = np.argsort(-a, kind="stable")            # descending
    inbox = m[order]

    fracs = {}
    for k in KS:
        t = max(1, int(round(k * n)))
        fracs[k] = float(inbox[:t].mean())

    hit = np.flatnonzero(inbox)
    best = float(hit[0] / n * 100.0) if hit.size else float("nan")

    pos = np.clip(a, 0, None)
    mass = float(pos[m].sum() / pos.sum() / area) if pos.sum() > 0 and area > 0 \
        else float("nan")
    return area, fracs, best, mass


def test_pool(loaders):
    ds = loaders.test.dataset
    base = ds.dataset if isinstance(ds, Subset) else ds
    while isinstance(base, Subset):
        base = base.dataset
    idxs = list(ds.indices) if isinstance(ds, Subset) else list(range(len(base.samples)))
    return base, idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run names")
    ap.add_argument("--methods", default="gradcam,hirescam,integrated_gradients",
                    help="post-hoc methods; ante-hoc is added automatically. "
                         "lime and kernelshap are ~4x slower.")
    ap.add_argument("--n-per-class", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--emit-tsv", default="figs/topk_localisation.tsv")
    a = ap.parse_args()

    if a.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    posthoc = [m for m in a.methods.split(",") if m]
    os.makedirs(os.path.dirname(a.emit_tsv) or ".", exist_ok=True)
    new = not os.path.exists(a.emit_tsv)
    tsv = open(a.emit_tsv, "a")
    if new:
        tsv.write("run\thead\tdataset\tclass\timage\tmethod\tarea\t"
                  + "\t".join(f"top{int(k*1000)/10:g}pct" for k in KS)
                  + "\tbest_rank_pct\tmass_conc\n")

    agg = collections.defaultdict(lambda: collections.defaultdict(list))

    for run in [r.strip() for r in a.runs.split(",") if r.strip()]:
        cfgp, ckpt = f"configs/generated/{run}.yaml", f"runs/{run}/best.pt"
        if not (os.path.exists(cfgp) and os.path.exists(ckpt)):
            print(f"SKIP {run}: missing config or checkpoint")
            continue
        cfg = load_config(cfgp)
        cfg["device"] = a.device
        dev = pick_device(cfg["device"])
        S, kind, root = cfg["data"]["img_size"], cfg["model"]["kind"], cfg["data"]["root"]
        lab = labels_for(root)
        if lab is None:
            print(f"SKIP {run}: no labels.json for {root}")
            continue
        ann = load_coco(lab)
        loaders = build_loaders(cfg)
        cfg["model"]["num_classes"] = len(loaders.classes)
        model = build_model(cfg).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
        model.eval()

        base, idxs = test_pool(loaders)
        cols = (["ours:" + kind] if kind != "blackbox" else []) + posthoc
        rng = np.random.default_rng(a.seed)
        print(f"\n{run}  ({kind}, {os.path.basename(root)}, {len(cols)} methods)",
              flush=True)

        for ci, cname in enumerate(loaders.classes):
            pool = [i for i in idxs if base.samples[i][1] == ci]
            if not pool:
                continue
            pick = [pool[j] for j in rng.choice(
                len(pool), min(a.n_per_class, len(pool)), replace=False)]
            for gi in pick:
                path, label = base.samples[gi]
                x, _ = base[gi]
                x = x.unsqueeze(0).to(dev)
                t = torch.tensor([label], device=dev)
                box = box_in_crop(path, ann, S, a.margin, True)
                if box is None or not box.any():
                    continue
                for name in cols:
                    try:
                        with torch.enable_grad():
                            at = (ante_hoc_attr(kind)(model, x, t)
                                  if name.startswith("ours:")
                                  else explain_posthoc(name, model, x, t)[0])
                        m = at.detach().float().cpu().numpy()
                        m = m[0, 0] if m.ndim == 4 else m.squeeze()
                        r = topk_metrics(m, box)
                        if r is None:
                            continue
                        area, fr, best, mass = r
                        tsv.write(f"{run}\t{kind}\t{os.path.basename(root)}\t{cname}\t"
                                  f"{os.path.basename(path)}\t{name}\t{area:.4f}\t"
                                  + "\t".join(f"{fr[k]:.4f}" for k in KS)
                                  + f"\t{best:.4f}\t{mass:.4f}\n")
                        key = (kind, name)
                        agg[key]["area"].append(area)
                        for k in KS:
                            agg[key][k].append(fr[k] / area if area > 0 else float("nan"))
                        if best == best:
                            agg[key]["best"].append(best)
                        if mass == mass:
                            agg[key]["mass"].append(mass)
                    except Exception as e:
                        print(f"    {name} failed on {os.path.basename(path)}: "
                              f"{type(e).__name__}", flush=True)
            tsv.flush()

    tsv.close()
    print(f"\nTSV -> {a.emit_tsv}")

    print(f"\n{'head':<11}{'method':<22}" + "".join(f"{'c@'+str(k*100)+'%':>9}" for k in KS)
          + f"{'mass':>8}{'1st hit':>9}{'n':>6}")
    print("-" * 92)
    for key in sorted(agg, key=lambda k: -st.mean(agg[k][0.01])):
        v = agg[key]
        print(f"{key[0]:<11}{key[1]:<22}"
              + "".join(f"{st.mean(v[k]):>9.2f}" for k in KS)
              + f"{st.mean(v['mass']):>8.2f}"
              + f"{st.mean(v['best']):>8.1f}%{len(v['mass']):>6}")

    print("""
c@k  = concentration among the top k% of pixels; 1.0 = no better than uniform
mass = conc_pos, the mass statistic reported in the technical report
1st hit = mean percentile rank of the FIRST pixel that lands inside the box;
          0.0% would mean the single highest-attribution pixel is always on the egg

  c@1% >> mass  -> §7.4 DISSOLVES. The explanation RANKS the egg highly and spreads its
                   mass. Deletion and conc_pos measure different properties and both
                   readings are correct. Rewrite §6 as "cannot draw a tight boundary",
                   not "does not find the egg".
  c@1% ~= mass  -> §7.4 is REAL. The explanation neither ranks nor locates the egg, and
                   the 0.0038 deletion score needs auditing.""")


if __name__ == "__main__":
    main()