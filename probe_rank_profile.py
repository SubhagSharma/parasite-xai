"""
probe_rank_profile.py — is the rank collapse real, or an artefact of one class and one
estimator?

THE FINDING BEING STRESS-TESTED
-------------------------------
A single measurement (class 0, one batch, participation ratio) gave:

    supervised MobileViT stage 4   rank  3.16 of 384   0.8% of channels
    supervised MobileViT stage 1   rank 20.78 of  48    43%
    DINOv2 self-supervised         rank 11.40 of 384   3.0%

The claim built on it is that **within-class representational rank bounds prototype
diversity**, which would explain four separate failed interventions and PDiscoFormer's
finding that self-supervised features let part-discovery priors be relaxed.

That is a lot of weight on one number. This probe checks it three ways.

  1. ALL CLASSES, not class 0. If the collapse is a property of Ascaris rather than of
     the network, the claim dies.
  2. THREE ESTIMATORS. "Effective rank" is not one thing:
       participation ratio  (sum L)^2 / sum L^2      — what was used so far
       stable rank          sum L / max L            — less sensitive to the tail
       entropy rank         exp(-sum p log p), p = L/sum L   — information-theoretic
     The qualitative ordering must hold under all three. If it flips, the finding is an
     estimator artefact.
  3. FULL TEST SET, not one batch. Enough patches that the covariance is estimated
     rather than guessed.

WHAT WOULD FALSIFY THE CLAIM
  * rank at stage 4 is high for some classes -> not a general property
  * the depth ordering reverses under stable rank or entropy rank -> estimator artefact
  * DINOv2 is not above supervised stage 4 under all three -> P1 fails

A NOTE ON WHAT P1 ACTUALLY SHOWED
DINOv2 (11.40) beats supervised stage 4 (3.16) as predicted, but supervised STAGE 1
beats DINOv2 outright (20.78). So "self-supervision preserves rank" is not the whole
story: rank falls with DEPTH in any network, and supervised training accelerates the
fall. Both axes need reporting, and this probe measures both.

    python -u probe_rank_profile.py --device cuda
    python -u probe_rank_profile.py --device cuda --max-batches 8   # faster
"""
from __future__ import annotations

import argparse
import collections
import math
import statistics as st

import torch


def rank_stats(S: torch.Tensor) -> dict:
    """Three effective-rank estimators from a centred (N, D) matrix."""
    S = S.float()
    S = S - S.mean(0, keepdim=True)
    ev = torch.linalg.svdvals(S) ** 2
    ev = ev[ev > 0]
    if ev.numel() == 0:
        return {"pr": float("nan"), "stable": float("nan"), "entropy": float("nan")}
    tot = ev.sum()
    p = ev / tot
    return {
        "pr": float(tot ** 2 / (ev ** 2).sum()),        # participation ratio
        "stable": float(tot / ev.max()),                # stable rank
        "entropy": float(torch.exp(-(p * p.log()).sum())),  # entropy rank
    }


def patches_by_class(feat, y, n_classes):
    """-> {class: (n_patches, D)} from a (B,D,H,W) feature map."""
    B, D, H, W = feat.shape
    zf = feat.permute(0, 2, 3, 1).reshape(B, H * W, D)
    out = {}
    for c in range(n_classes):
        m = (y == c)
        if m.any():
            out[c] = zf[m.to(zf.device)].reshape(-1, D)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="roi477_protopnet_120ep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = whole test set")
    ap.add_argument("--max-patches", type=int, default=20000,
                    help="cap per class per stage; SVD cost is O(N D^2)")
    ap.add_argument("--skip-dino", action="store_true")
    a = ap.parse_args()

    import timm
    from pxai.utils import load_config, pick_device
    from pxai.data import build_loaders
    from pxai.models import build_model

    cfg = load_config(f"configs/generated/{a.run}.yaml")
    cfg["device"] = a.device
    dev = pick_device(cfg["device"])
    loaders = build_loaders(cfg)
    classes = loaders.classes
    cfg["model"]["num_classes"] = len(classes)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(f"runs/{a.run}/best.pt", map_location=dev)["model"])
    model.eval()
    net = model.backbone.net

    dino = None
    if not a.skip_dino:
        dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                                 num_classes=0, img_size=224).to(dev).eval()

    # accumulate patches per (source, stage, class)
    acc = collections.defaultdict(list)
    n_batches = 0
    with torch.no_grad():
        for x, y in loaders.test:
            x = x.to(dev)
            for si, f in enumerate(net(x)):
                for c, p in patches_by_class(f, y, len(classes)).items():
                    key = (f"mobilevit_s{si}", c)
                    if sum(t.shape[0] for t in acc[key]) < a.max_patches:
                        acc[key].append(p.cpu())
            if dino is not None:
                t = dino.forward_features(x)[:, 1:]
                n = int(round(math.sqrt(t.shape[1])))
                f = t.transpose(1, 2).reshape(t.shape[0], -1, n, n)
                for c, p in patches_by_class(f, y, len(classes)).items():
                    key = ("dinov2", c)
                    if sum(t2.shape[0] for t2 in acc[key]) < a.max_patches:
                        acc[key].append(p.cpu())
            n_batches += 1
            if a.max_batches and n_batches >= a.max_batches:
                break

    print(f"\n{n_batches} batches, {len(classes)} classes\n")

    sources = sorted({k[0] for k in acc},
                     key=lambda s: (s != "dinov2", s))
    rows = {}
    for src in sources:
        per_class = collections.defaultdict(list)
        D = None
        for c in range(len(classes)):
            key = (src, c)
            if key not in acc:
                continue
            S = torch.cat(acc[key], 0)[: a.max_patches]
            D = S.shape[1]
            r = rank_stats(S)
            for k, v in r.items():
                per_class[k].append(v)
        if D is None:
            continue
        rows[src] = {k: (st.mean(v), min(v), max(v)) for k, v in per_class.items()}
        rows[src]["D"] = D

    print(f"{'source':<18}{'ch':>5}{'PR mean':>10}{'PR min':>8}{'PR max':>8}"
          f"{'stable':>9}{'entropy':>9}{'PR/ch':>8}")
    print("-" * 75)
    for src in sources:
        if src not in rows:
            continue
        r = rows[src]
        pr, lo, hi = r["pr"]
        print(f"{src:<18}{r['D']:>5}{pr:>10.2f}{lo:>8.2f}{hi:>8.2f}"
              f"{r['stable'][0]:>9.2f}{r['entropy'][0]:>9.2f}{pr / r['D']:>7.1%}")

    print("""
READING
  PR min / PR max  spread across the 11 classes. A tight spread means the collapse is a
                   property of the NETWORK; a wide one means it is a property of some
                   classes and the claim does not generalise.
  stable, entropy  the ordering across sources must match participation ratio. If it
                   flips, the finding is an estimator artefact and must be withdrawn.
  PR/ch            rank as a fraction of available channels -- the fair comparison,
                   since a 384-channel layer has more room than a 48-channel one.

TWO AXES, NOT ONE
  Depth destroys rank in ANY network (mobilevit_s0 -> s4 falls steeply).
  Supervised training destroys MORE at matched depth (dinov2 > mobilevit_s4).
  Both must be stated; P1 only established the second.""")


if __name__ == "__main__":
    main()
