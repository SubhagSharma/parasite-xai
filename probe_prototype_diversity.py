"""
probe_prototype_diversity.py — how redundant are the prototypes, really?

WHY MEASURE BEFORE FIXING
-------------------------
The observation motivating this is visual: in figs/protograd_overview.png the three
highest-weighted prototypes of a class have near-identical weights (1.23, 1.19, 1.00)
and near-identical gradient attribution maps. Ideally one would find the shell, another
the interior, another the polar plug.

"Near-identical" is not a number. Without one there is no way to tell whether an
intervention helped, and no way to report that it did.

WHY REDUNDANCY IS EXPECTED, FROM THE LOSS
-----------------------------------------
pxai/models/protopnet.py:

    cluster = -(sim_pooled * oh).max(1).values.mean()     # max over prototypes
    sep     =  (sim_pooled * (1 - oh)).max(1).values.mean()
    l1      =  (self.last.weight * off).abs().sum()

**No term depends on two prototypes jointly.** Cluster is a max, so only the single
best-matching prototype of the correct class receives gradient. Separation is a max.
L1 is per-weight. Nothing pushes prototypes apart, so any diversity is accidental — and
prototype push then projects all k of them onto whatever tight region the embedding has
collapsed same-class patches into.

WHAT IS MEASURED
----------------
Five statistics, each answering a different sense of "the same".

  cos          mean pairwise cosine similarity between same-class prototypes.
               1.00 = identical direction, 0.00 = orthogonal. Because add_on ends in
               Sigmoid, embeddings are non-negative, so cos = 0 requires DISJOINT
               CHANNEL SUPPORT — i.e. the prototypes respond to different feature
               dimensions, which is what "different visual structure" means physically.

  eff          effective number of prototypes per class, from the participation ratio
               of the eigenvalues of the k x k Gram matrix:
                   eff = (sum lambda)^2 / sum(lambda^2)
               k means fully independent; 1.0 means all k prototypes span a single
               direction and the class effectively has ONE prototype.

  jacc         mean pairwise Jaccard overlap of each prototype's top-32 channels. A
               direct read of "do they use the same features".

  map_corr     mean pairwise Pearson correlation between the per-prototype GRADIENT
               attribution maps (the Part II attribution, at pixel resolution). This is
               the one that matters most: it measures whether two prototypes highlight
               the same structure in the image, which is exactly the visual observation.

  cells        number of DISTINCT argmax cells the class's prototypes select, pooled
               over the test set, out of h*w. If all k prototypes always fire on the
               same cell, this is 1.

A NOTE ON WHAT "GOOD" LOOKS LIKE
--------------------------------
Not zero overlap. The prototypes of one class should all be somewhere on the egg — the
egg occupies 1-2 cells of a 7x7 grid, so forcing them onto DIFFERENT CELLS would push
four of five onto background and destroy localisation.

The target is diversity WITHIN the object: same region, different structures, therefore
different channels and different pixel-level maps. So the statistics to move are `cos`,
`jacc` and `map_corr`; `cells` should stay small, and a large jump in `cells` alongside
a drop in localisation is the failure mode to watch for.

    python -u probe_prototype_diversity.py --device cuda \\
        --runs roi477_protopnet_120ep,roi477_protopnet_s2337_120ep
"""
from __future__ import annotations

import argparse
import collections
import itertools
import os
import statistics as st

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model


def class_w(head, target):
    w = head.last.weight
    if getattr(head, "pip_sparsity", False):
        w = F.relu(w)
    return w[target]


def proto_grad_map(model, x, target, p):
    """Pixel-resolution attribution for ONE prototype (Part II method)."""
    head = model.head
    xi = x.clone().detach().requires_grad_(True)
    if hasattr(head, "stages"):                    # multi-scale head
        from pxai.models.protopnet_multiscale import pyramid_forward
        sp, _ = head._similarities(pyramid_forward(model.backbone, xi))
    else:
        sp, _ = head._similarities(model.backbone(xi))
    g, = torch.autograd.grad(sp[:, p].sum(), xi)
    return (g * xi).abs().sum(1).detach()[0]                 # (H,W)


def pairwise_mean(vals):
    return float(st.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--n-per-class", type=int, default=3,
                    help="images per class for the map-correlation and cell statistics")
    ap.add_argument("--top-channels", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--emit-tsv", default="figs/prototype_diversity.tsv")
    a = ap.parse_args()

    if a.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    os.makedirs(os.path.dirname(a.emit_tsv) or ".", exist_ok=True)
    new = not os.path.exists(a.emit_tsv)
    tsv = open(a.emit_tsv, "a")
    if new:
        tsv.write("run\tclass\tk\tcos\teff\tjacc\tmap_corr\tcells\tcells_max\n")

    for run in [r.strip() for r in a.runs.split(",") if r.strip()]:
        cfgp, ckpt = f"configs/generated/{run}.yaml", f"runs/{run}/best.pt"
        if not (os.path.exists(cfgp) and os.path.exists(ckpt)):
            print(f"SKIP {run}: missing config or checkpoint")
            continue
        cfg = load_config(cfgp)
        cfg["device"] = a.device
        dev = pick_device(cfg["device"])
        if cfg["model"]["kind"] not in ("protopnet", "protopnet_diverse",
                                        "protopnet_ms"):
            print(f"SKIP {run}: not a prototype head")
            continue
        loaders = build_loaders(cfg)
        classes = loaders.classes
        cfg["model"]["num_classes"] = len(classes)
        model = build_model(cfg).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
        model.eval()

        head = model.head
        P = head.prototypes.shape[0]
        C = len(classes)
        k = P // C
        pv = head.prototypes.detach().flatten(1)              # (P,D)
        print(f"\n{run}: {P} prototypes, {k} per class, dim {pv.shape[1]}")

        # ---- embedding-space statistics, no data needed -------------------------
        pn = F.normalize(pv, dim=1)
        topch = pv.topk(min(a.top_channels, pv.shape[1]), dim=1).indices

        # ---- data-dependent statistics ------------------------------------------
        ds = loaders.test.dataset
        base = ds.dataset if isinstance(ds, Subset) else ds
        while isinstance(base, Subset):
            base = base.dataset
        idxs = list(ds.indices) if isinstance(ds, Subset) else range(len(base.samples))
        rng = np.random.default_rng(a.seed)

        rows = []
        for ci, cname in enumerate(classes):
            ps = list(range(ci * k, (ci + 1) * k))
            if len(ps) < 2:
                continue

            cos = pairwise_mean([float(pn[p] @ pn[q]) for p, q in
                                 itertools.combinations(ps, 2)])
            G = (pn[ps] @ pn[ps].t()).cpu().numpy()
            ev = np.clip(np.linalg.eigvalsh(G), 0, None)
            eff = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))
            jacc = pairwise_mean([
                len(set(topch[p].tolist()) & set(topch[q].tolist())) /
                len(set(topch[p].tolist()) | set(topch[q].tolist()))
                for p, q in itertools.combinations(ps, 2)])

            pool = [i for i in idxs if base.samples[i][1] == ci]
            corrs, cells = [], set()
            hw = None
            for gi in [pool[j] for j in rng.choice(
                    len(pool), min(a.n_per_class, len(pool)), replace=False)]:
                x, _ = base[gi]
                x = x.unsqueeze(0).to(dev)
                t = torch.tensor([ci], device=dev)
                with torch.no_grad():
                    if hasattr(head, "stages"):
                        from pxai.models.protopnet_multiscale import pyramid_forward
                        _, per_stage = head._similarities(pyramid_forward(model.backbone, x))
                        # concatenate stages so the argmax-cell statistic still works;
                        # cells from different stages are distinct by construction
                        sim = torch.cat([F.interpolate(m, size=per_stage[0].shape[-2:],
                                                       mode="nearest")
                                         for m in per_stage], dim=1)
                    else:
                        _, sim = head._similarities(model.backbone(x))
                hw = sim.shape[-2:]
                flat = sim.reshape(1, P, -1)
                for p in ps:
                    cells.add(int(flat[0, p].argmax()))
                maps = {}
                for p in ps:
                    maps[p] = proto_grad_map(model, x, t, p).flatten().cpu().numpy()
                for p, q in itertools.combinations(ps, 2):
                    v = np.corrcoef(maps[p], maps[q])[0, 1]
                    if v == v:
                        corrs.append(float(v))

            mc = pairwise_mean(corrs)
            nc = hw[0] * hw[1] if hw else 0
            rows.append((cname, cos, eff, jacc, mc, len(cells), nc))
            tsv.write(f"{run}\t{cname}\t{k}\t{cos:.4f}\t{eff:.4f}\t{jacc:.4f}\t"
                      f"{mc:.4f}\t{len(cells)}\t{nc}\n")
        tsv.flush()

        print(f"\n{'class':<28}{'cos':>7}{'eff':>7}{'jacc':>7}{'map_corr':>10}{'cells':>8}")
        print("-" * 68)
        for cname, cos, eff, jacc, mc, nc_used, nc in rows:
            print(f"{cname[:26]:<28}{cos:>7.3f}{eff:>7.2f}{jacc:>7.3f}{mc:>10.3f}"
                  f"{nc_used:>5}/{nc}")
        print("-" * 68)
        print(f"{'MEAN':<28}{st.mean([r[1] for r in rows]):>7.3f}"
              f"{st.mean([r[2] for r in rows]):>7.2f}"
              f"{st.mean([r[3] for r in rows]):>7.3f}"
              f"{st.mean([r[4] for r in rows]):>10.3f}"
              f"{st.mean([r[5] for r in rows]):>8.1f}")

    tsv.close()
    print(f"""
TSV -> {a.emit_tsv}

INTERPRETATION   k prototypes per class
  cos      1.00 = identical direction, 0.00 = orthogonal (and, because embeddings are
           non-negative after the Sigmoid add_on, orthogonal => DISJOINT channels)
  eff      effective independent prototypes. k = fully independent, 1.0 = the class
           really has one prototype wearing k hats
  jacc     top-channel overlap. 1.00 = same features, 0.00 = disjoint features
  map_corr correlation between per-prototype PIXEL-LEVEL attribution maps. This is the
           statistic that matches the visual observation. > 0.8 = the prototypes
           highlight the same structure
  cells    distinct argmax cells used, out of h*w

WHAT TO AIM FOR
  DOWN:  cos, jacc, map_corr        UP: eff
  STABLE: cells -- the egg covers 1-2 cells of a 7x7 grid, so forcing prototypes onto
          different CELLS would push most of them onto background and destroy
          localisation. The goal is diversity WITHIN the object: same region, different
          structures, therefore different channels and different pixel maps.
  A large jump in `cells` together with a fall in conc_pos is the failure mode.""")


if __name__ == "__main__":
    main()
