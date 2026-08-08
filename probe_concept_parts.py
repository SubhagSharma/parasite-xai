"""
probe_concept_parts.py — does each concept slot find the feature it names?

THREE MEASUREMENTS
==================
The pooled figure gave concentration 6.03 when a concept is TRUE for the species and
4.83 when FALSE (uniform = 1.0). Both are well-localised; the gap is only 25%. That
supports "each concept has a localised evidence region" but NOT "A_operculum finds the
operculum". These three measurements separate those claims.

--- 1. PER-CONCEPT true/false gap ---------------------------------------------
The pooled number hides variation. A concept present in 3 of 11 classes has far more
discriminative signal than one present in 6, so the gap should be much larger for
high-contrast concepts. If `operculum` and `has_polar_plugs` separate cleanly while
`shell_texture=smooth` does not, the claim can be made FOR THOSE CONCEPTS and qualified
for the rest -- which is a real result rather than a hedge.

--- 2. CENTROID CONSISTENCY ---------------------------------------------------
Concentration says the map is on the egg. It does not say it is on the same PART of the
egg across images. Centroids are computed in BOX-RELATIVE coordinates -- (0,0) is the
top-left of the annotation box, (1,1) the bottom-right -- so this measures where on the
egg the slot attends, not where the egg happens to sit in the frame.

    spread near 0.1  -> the slot lands on the same anatomical location every time. That
                        is part discovery in the PDiscoNet sense (their equivariance
                        prior targets exactly this).
    spread near 0.3  -> the slot is on the egg but wandering; it has found the object,
                        not the part.

--- 3. ATTENTION vs GRADIENT read-out -----------------------------------------
`self.attn` is a Conv2d over the (B, 384, 7, 7) feature map, so the attention maps carry
**exactly the same 7x7 resolution bottleneck as ProtoPNet's similarity maps** -- one cell
is 2% of the frame. That is why they render as smooth blobs.

Part II showed that taking the gradient of the head's own interpretable intermediate
with respect to the INPUT escapes that limit: c@1% improved 5.4x for ProtoPNet, 5.6x for
CBM, 2.2x for B-cos, all converging on the receptive-field ceiling. The same trick
applies here, with the concept logit as the intermediate:

    A_k(x) = | d c_logit[k] / dx | . x        at pixel resolution

and it stays per-concept, so the named claim survives.

    prediction: gradient concentration > attention concentration for every concept,
    by roughly the Part II factor. If not, the concept head's bottleneck is not the
    grid, and Part II's mechanism does not generalise to attention-based heads.

    python -u probe_concept_parts.py --run roi477_parts_120ep --device cuda
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
from pxai.concepts_loader import load_concept_table
from pxai.eval.cropgeom import load_coco, box_in_crop


def conc(a, mask):
    a = np.clip(np.asarray(a, dtype=np.float64), 0, None)
    tot = a.sum()
    ar = float(mask.mean())
    return float(a[mask].sum() / tot / ar) if tot > 0 and ar > 0 else float("nan")


def centroid_in_box(a, mask):
    """Attention centroid in BOX-RELATIVE coordinates.

    (0,0) is the top-left of the annotation box, (1,1) the bottom-right. Without this
    normalisation the spread would measure where the EGG sits in the frame rather than
    where the slot attends on the egg.
    """
    a = np.clip(np.asarray(a, dtype=np.float64), 0, None)
    if a.sum() <= 0:
        return None
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if y1 <= y0 or x1 <= x0:
        return None
    H, W = a.shape
    gy, gx = np.mgrid[0:H, 0:W]
    cy = float((a * gy).sum() / a.sum())
    cx = float((a * gx).sum() / a.sum())
    return ((cy - y0) / (y1 - y0), (cx - x0) / (x1 - x0))


def grad_concept_map(model, x, k):
    """|d c_logit[k] / dx| . x at PIXEL resolution -- the Part II read-out, per concept."""
    xi = x.clone().detach().requires_grad_(True)
    feat = model.features(xi)
    _, c_logit = model.head(feat)
    g, = torch.autograd.grad(c_logit[0, k], xi)
    return (g * xi).abs().sum(1)[0].detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="roi477_parts_120ep")
    ap.add_argument("--n", type=int, default=8, help="images per class")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--skip-grad", action="store_true",
                    help="skip measurement 3 (one backward per concept per image)")
    ap.add_argument("--emit-tsv", default="figs/concept_parts_eval.tsv")
    a = ap.parse_args()

    cfg = load_config(f"configs/generated/{a.run}.yaml")
    cfg["device"] = a.device
    dev = pick_device(cfg["device"])
    S, root = cfg["data"]["img_size"], cfg["data"]["root"]
    lab = os.path.join(root, "labels.json")
    if not os.path.exists(lab):
        lab = os.path.join(os.path.dirname(root.rstrip("/")),
                           "Chula-ParasiteEgg-11", "labels.json")
    ann = load_coco(lab)
    loaders = build_loaders(cfg)
    classes = loaders.classes
    cfg["model"]["num_classes"] = len(classes)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(f"runs/{a.run}/best.pt", map_location=dev)["model"])
    model.eval()
    head = model.head
    table, names = load_concept_table(
        cfg["model"]["concept_parts"]["concepts_csv"], classes)
    K = len(names)

    ds = loaders.test.dataset
    base = ds.dataset if isinstance(ds, Subset) else ds
    while isinstance(base, Subset):
        base = base.dataset
    idxs = list(ds.indices) if isinstance(ds, Subset) else list(range(len(base.samples)))
    rng = np.random.default_rng(a.seed)

    attn_pos = collections.defaultdict(list)
    attn_neg = collections.defaultdict(list)
    grad_pos = collections.defaultdict(list)
    cent = collections.defaultdict(list)          # (concept, class) -> centroids

    os.makedirs(os.path.dirname(a.emit_tsv) or ".", exist_ok=True)
    tsv = open(a.emit_tsv, "w")
    tsv.write("run\tclass\timage\tconcept\ttrue\tattn_conc\tgrad_conc\tcy\tcx\n")

    for ci, cname in enumerate(classes):
        pool = [i for i in idxs if base.samples[i][1] == ci]
        if not pool:
            continue
        pick = [pool[j] for j in rng.choice(len(pool), min(a.n, len(pool)),
                                            replace=False)]
        for gi in pick:
            path, y = base.samples[gi]
            x, _ = base[gi]
            x = x.unsqueeze(0).to(dev)
            box = box_in_crop(path, ann, S, a.margin, True)
            if box is None or not box.any():
                continue
            with torch.no_grad():
                feat = model.features(x)
            for k in range(K):
                am = head.concept_map(feat, k, size=x.shape[-2:])
                am = am.detach().float().cpu().numpy()[0, 0]
                ac = conc(am, box)
                truth = int(table[y, k])
                (attn_pos if truth else attn_neg)[k].append(ac)

                gc = float("nan")
                if not a.skip_grad:
                    gm = grad_concept_map(model, x, k)
                    gc = conc(gm, box)
                    if truth:
                        grad_pos[k].append(gc)

                cxy = centroid_in_box(am, box)
                if cxy and truth:
                    cent[(k, ci)].append(cxy)
                tsv.write(f"{a.run}\t{cname}\t{os.path.basename(path)}\t{names[k]}\t"
                          f"{truth}\t{ac:.4f}\t{gc:.4f}\t"
                          f"{cxy[0]:.4f}\t{cxy[1]:.4f}\n" if cxy else
                          f"{a.run}\t{cname}\t{os.path.basename(path)}\t{names[k]}\t"
                          f"{truth}\t{ac:.4f}\t{gc:.4f}\tnan\tnan\n")
        tsv.flush()
    tsv.close()

    npos = table.sum(0).numpy()

    # ---- 1. per-concept gap --------------------------------------------------
    print(f"\n=== 1. PER-CONCEPT: does the slot localise better when the feature is "
          f"PRESENT? ===")
    print(f"{'concept':<32}{'classes+':>9}{'true':>8}{'false':>8}{'gap':>8}"
          f"{'grad':>8}")
    rows = []
    for k in range(K):
        p = st.mean(attn_pos[k]) if attn_pos[k] else float("nan")
        n = st.mean(attn_neg[k]) if attn_neg[k] else float("nan")
        g = st.mean(grad_pos[k]) if grad_pos[k] else float("nan")
        rows.append((k, p, n, p - n if p == p and n == n else float("nan"), g))
    for k, p, n, d, g in sorted(rows, key=lambda r: -(r[3] if r[3] == r[3] else -9)):
        print(f"{names[k][:30]:<32}{int(npos[k]):>6}/11{p:>8.2f}{n:>8.2f}{d:>8.2f}"
              f"{g:>8.2f}")

    ok = [r for r in rows if r[3] == r[3]]
    print(f"\n  mean gap {st.mean([r[3] for r in ok]):+.2f}   "
          f"concepts with gap > 1.0: {sum(1 for r in ok if r[3] > 1.0)}/{K}")
    print("  a large gap on HIGH-CONTRAST concepts (operculum, has_polar_plugs) with a")
    print("  small one elsewhere means the claim holds for those concepts specifically.")

    # ---- 2. centroid consistency ---------------------------------------------
    print(f"\n=== 2. CENTROID CONSISTENCY: same anatomical spot across images? ===")
    print("  spread is the std of the box-relative centroid; 0 = identical location")
    print(f"{'concept':<32}{'species':<24}{'spread':>8}{'n':>5}")
    spreads = []
    for (k, ci), pts in sorted(cent.items(), key=lambda kv: -len(kv[1])):
        if len(pts) < 4:
            continue
        arr = np.array(pts)
        s = float(np.sqrt(arr.std(0).mean() ** 2 * 2))
        spreads.append(s)
        if len(spreads) <= 15:
            print(f"{names[k][:30]:<32}{classes[ci][:22]:<24}{s:>8.3f}{len(pts):>5}")
    if spreads:
        print(f"\n  mean spread {st.mean(spreads):.3f} over "
              f"{len(spreads)} (concept, species) pairs")
        print("  < 0.15 -> genuine part discovery: the same location every time")
        print("  > 0.30 -> on the egg but wandering; the object, not the part")

    # ---- 3. attention vs gradient --------------------------------------------
    if not a.skip_grad:
        ap_ = [v for k in range(K) for v in attn_pos[k]]
        gp_ = [v for k in range(K) for v in grad_pos[k]]
        if ap_ and gp_:
            print(f"\n=== 3. READ-OUT: attention (7x7 grid) vs gradient (pixel) ===")
            print(f"  attention  {st.mean(ap_):>6.2f}")
            print(f"  gradient   {st.mean(gp_):>6.2f}   "
                  f"({st.mean(gp_) / max(st.mean(ap_), 1e-9):.2f}x)")
            print("""
  self.attn is a Conv2d over the 7x7 feature map, so the attention maps carry the SAME
  resolution bottleneck as ProtoPNet's similarity maps. Part II showed the gradient
  read-out escapes it (5.4x for ProtoPNet, 5.6x CBM, 2.2x B-cos).
    gradient >> attention -> the mechanism generalises to attention-based heads, and
                             the concept maps should be reported from the gradient
    gradient ~= attention -> the concept head's limit is not the grid; Part II's
                             mechanism does not transfer""")
    print(f"\nTSV -> {a.emit_tsv}")


if __name__ == "__main__":
    main()
