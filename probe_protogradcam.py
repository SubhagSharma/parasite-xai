"""
probe_protogradcam.py — a localizable ProtoPNet explanation.

THE PROBLEM WITH THE CURRENT EXPLANATION
----------------------------------------
ProtoPNet explains a prediction with `sim_maps`, a 7x7 grid per prototype, reduced by
max-pool to P single cells and upsampled to 224. One cell is 2.0% of the frame. Three
interventions were tried and all failed to fix localisation:

    corrected attribution (argmax + learned weights)   conc_pos 1.08 -> 2.53
    egg-constrained prototype push                     3.08 -> 2.45  (WORSE)
    448px input, 14x14 grid                            2.53 -> 1.61  (WORSE, -6 pts acc)

All three stayed *inside* the 7x7 feature grid. They changed which cell, or how many
cells. None left the grid.

THE EVIDENCE THAT LOCALISATION IS ACHIEVABLE
--------------------------------------------
On the SAME checkpoint and the SAME images:

    Integrated Gradients   mass 5.31   c@1% 12.40   1st hit 0.0%
    ours:protopnet         mass 2.53   c@1%  2.26   1st hit 4.8%

A gradient method extracts roughly twice the localisation from the same model. The
backbone knows where the egg is; the prototype grid does not expose it.

THE PROPOSAL
------------
Explain each prototype's contribution by its gradient with respect to the INPUT, at
pixel resolution, instead of by its 7x7 similarity map:

    A_p(x) = |d sim_pooled[p] / dx| . x           (pixel resolution)
    A(x)   = sum_p W[y,p] * A_p(x)                W = relu(last.weight) under pip_sparsity

This is NOT Integrated Gradients. IG explains the logit as one undifferentiated
quantity. This explains it PER PROTOTYPE, so the claim remains

    "this region looks like prototype 3, which is a Trichuris polar plug"

which is the entire point of a case-based model. The architecture is unchanged, the
bottleneck is unchanged, the explanation is still prototype-mediated. Only the
VISUALISATION leaves the 49-cell grid.

Because sim_pooled is a max over space, d sim_pooled[p]/dx is non-zero only through the
argmax cell's receptive field — so A_p is automatically localised to the region that
prototype actually matched, at whatever resolution the backbone's receptive field
supports rather than at 32px granularity.

VARIANTS COMPARED
    ours:protopnet   the current corrected attribution (baseline, expect ~2.53)
    protograd        |d sim_pooled[p]/dx| . x, weighted, summed         <- the proposal
    protograd_pos    same, positive part only
    protograd_smooth SmoothGrad over 8 noisy copies (sigma 0.10), variance reduction
    per_prototype    the top-1 prototype's map alone, to check the per-prototype claim
                     survives (a good aggregate could hide unlocalised individuals)

PREDICTION, RECORDED BEFORE THE RUN
    mass >= 4, c@1% >= 8. Below IG (5.31 / 12.40) because it is constrained to the
    prototype path, but well clear of 2.53. If it lands at ~2.5 this is a FOURTH
    rejected mechanism and the negative result is close to airtight.

No retraining. Existing checkpoints, ~30 min for five seeds.

    python -u probe_protogradcam.py \
        --runs roi477_protopnet_120ep,roi477_protopnet_s2337_120ep \
        --device cuda
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics as st

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.eval.cropgeom import load_coco, box_in_crop

VARIANTS = ("ours:protopnet", "protograd", "protograd_pos",
            "protograd_smooth", "per_prototype")


def class_w(head, target):
    w = head.last.weight
    if getattr(head, "pip_sparsity", False):
        w = F.relu(w)
    return w[target]                                        # (B,P)


def baseline_attr(model, x, target):
    """The current corrected attribution: argmax cells, learned weights, upsampled."""
    head = model.head
    with torch.no_grad():
        _, sim = head._similarities(model.backbone(x))
        B, P, h, w = sim.shape
        wy = class_w(head, target).view(B, P, 1, 1)
        flat = sim.reshape(B, P, h * w)
        mx, idx = flat.max(-1)
        sparse = torch.zeros_like(flat)
        sparse.scatter_(2, idx.unsqueeze(-1), (mx * wy.view(B, P)).unsqueeze(-1))
        a = sparse.view(B, P, h, w).sum(1, keepdim=True)
    return F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)


def proto_grad(model, x, target, per_proto=False, noise=0.0, n_noise=1):
    """Gradient of each prototype's pooled similarity wrt the INPUT, pixel resolution.

    Returns (B,1,H,W) summed over prototypes and weighted by W[y,p], or the single
    highest-weighted prototype's map when per_proto is set.
    """
    head = model.head
    B = x.shape[0]
    with torch.no_grad():
        _, sim0 = head._similarities(model.backbone(x))
        P = sim0.shape[1]
        wy = class_w(head, target)                          # (B,P)
    keep = [int(wy[0].argmax())] if per_proto else \
        [p for p in range(P) if float(wy[0, p]) > 0]
    if not keep:
        keep = list(range(P))

    total = torch.zeros_like(x[:, :1])
    for _ in range(max(1, n_noise)):
        xi = x if noise <= 0 else x + torch.randn_like(x) * noise
        xi = xi.clone().detach().requires_grad_(True)
        sp, _ = head._similarities(model.backbone(xi))       # (B,P)
        # one backward over the weighted sum of the kept prototypes; the max-pool means
        # each prototype's gradient flows only through its own argmax receptive field,
        # so the sum stays a per-prototype decomposition rather than a blend
        obj = (sp[:, keep] * wy[:, keep]).sum()
        g, = torch.autograd.grad(obj, xi)
        total = total + (g * xi).abs().sum(1, keepdim=True).detach()
    return total / max(1, n_noise)


def attributions(model, x, target):
    out = {"ours:protopnet": baseline_attr(model, x, target)}
    g = proto_grad(model, x, target)
    out["protograd"] = g
    out["protograd_pos"] = torch.clamp(g, min=0)
    out["protograd_smooth"] = proto_grad(model, x, target, noise=0.10, n_noise=8)
    out["per_prototype"] = proto_grad(model, x, target, per_proto=True)
    return out


def metrics(a, mask):
    a = np.asarray(a, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    area = float(m.mean())
    if area <= 0 or not np.isfinite(a).all():
        return (float("nan"),) * 4
    pos = np.clip(a, 0, None)
    mass = float(pos[m].sum() / pos.sum() / area) if pos.sum() > 0 else float("nan")
    order = np.argsort(-a, kind="stable")
    inbox = m[order]
    t = max(1, int(round(0.01 * a.size)))
    c1 = float(inbox[:t].mean()) / area
    peak = float(inbox[0])
    hit = np.flatnonzero(inbox)
    first = float(hit[0] / a.size * 100.0) if hit.size else float("nan")
    return mass, c1, peak, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="roi477_protopnet_120ep")
    ap.add_argument("--n-per-class", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--emit-tsv", default="figs/protogradcam.tsv")
    a = ap.parse_args()

    if a.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    os.makedirs(os.path.dirname(a.emit_tsv) or ".", exist_ok=True)
    new = not os.path.exists(a.emit_tsv)
    tsv = open(a.emit_tsv, "a")
    if new:
        tsv.write("run\tclass\timage\tvariant\tarea\tmass\tc1pct\tpeak\tfirst_hit_pct\n")

    agg = collections.defaultdict(lambda: collections.defaultdict(list))

    for run in [r.strip() for r in a.runs.split(",") if r.strip()]:
        cfgp, ckpt = f"configs/generated/{run}.yaml", f"runs/{run}/best.pt"
        if not (os.path.exists(cfgp) and os.path.exists(ckpt)):
            print(f"SKIP {run}: missing config or checkpoint")
            continue
        cfg = load_config(cfgp)
        cfg["device"] = a.device
        dev = pick_device(cfg["device"])
        if cfg["model"]["kind"] != "protopnet":
            print(f"SKIP {run}: not a protopnet head")
            continue
        S, root = cfg["data"]["img_size"], cfg["data"]["root"]
        lab = os.path.join(root, "labels.json")
        if not os.path.exists(lab):
            lab = os.path.join(os.path.dirname(root.rstrip("/")),
                               "Chula-ParasiteEgg-11", "labels.json")
        ann = load_coco(lab)
        loaders = build_loaders(cfg)
        cfg["model"]["num_classes"] = len(loaders.classes)
        model = build_model(cfg).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
        model.eval()

        ds = loaders.test.dataset
        base = ds.dataset if isinstance(ds, Subset) else ds
        while isinstance(base, Subset):
            base = base.dataset
        idxs = list(ds.indices) if isinstance(ds, Subset) else range(len(base.samples))
        rng = np.random.default_rng(a.seed)
        print(f"\n{run}  (grid {S // 32}x{S // 32}, "
              f"{model.head.prototypes.shape[0]} prototypes)", flush=True)

        for ci, cname in enumerate(loaders.classes):
            pool = [i for i in idxs if base.samples[i][1] == ci]
            if not pool:
                continue
            for gi in [pool[j] for j in rng.choice(
                    len(pool), min(a.n_per_class, len(pool)), replace=False)]:
                path, label = base.samples[gi]
                x, _ = base[gi]
                x = x.unsqueeze(0).to(dev)
                t = torch.tensor([label], device=dev)
                box = box_in_crop(path, ann, S, a.margin, True)
                if box is None or not box.any():
                    continue
                try:
                    maps = attributions(model, x, t)
                except Exception as e:
                    print(f"    failed on {os.path.basename(path)}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                for v, mp in maps.items():
                    arr = mp.detach().float().cpu().numpy()[0, 0]
                    mass, c1, pk, fh = metrics(arr, box)
                    if mass != mass:
                        continue
                    tsv.write(f"{run}\t{cname}\t{os.path.basename(path)}\t{v}\t"
                              f"{box.mean():.4f}\t{mass:.4f}\t{c1:.4f}\t{pk:.0f}\t"
                              f"{fh:.4f}\n")
                    agg[(run, v)]["mass"].append(mass)
                    agg[(run, v)]["c1"].append(c1)
                    agg[(run, v)]["peak"].append(pk)
                    agg[(run, v)]["first"].append(fh)
            tsv.flush()

    tsv.close()
    print(f"\nTSV -> {a.emit_tsv}")
    print(f"\n{'run':<34}{'variant':<19}{'mass':>7}{'c@1%':>8}{'peak%':>7}"
          f"{'1st hit':>9}{'n':>5}")
    print("-" * 89)
    for k in sorted(agg, key=lambda k: (k[0], VARIANTS.index(k[1])
                                        if k[1] in VARIANTS else 9)):
        v = agg[k]
        print(f"{k[0]:<34}{k[1]:<19}{st.mean(v['mass']):>7.2f}{st.mean(v['c1']):>8.2f}"
              f"{st.mean(v['peak'])*100:>6.0f}%{st.mean(v['first']):>8.1f}%"
              f"{len(v['mass']):>5}")

    # seed-averaged, which is what should be reported
    per_v = collections.defaultdict(lambda: collections.defaultdict(list))
    for (run, v), d in agg.items():
        for m in ("mass", "c1", "first"):
            per_v[v][m] += d[m]
    print(f"\nSEED-AVERAGED\n{'variant':<19}{'mass':>7}{'c@1%':>8}{'1st hit':>9}{'n':>6}")
    print("-" * 49)
    for v in VARIANTS:
        if v not in per_v:
            continue
        d = per_v[v]
        print(f"{v:<19}{st.mean(d['mass']):>7.2f}{st.mean(d['c1']):>8.2f}"
              f"{st.mean(d['first']):>8.1f}%{len(d['mass']):>6}")

    print("""
REFERENCE   1.0 = random | 16.3 = perfect | IG on this model: mass 5.31, c@1% 12.40
            current ProtoPNet explanation: mass 2.53, c@1% 2.26

READING
  protograd mass >= 4 and c@1% >= 8
      -> A LOCALIZABLE PROTOTYPE EXPLANATION. The architecture is unchanged and the
         claim stays per-prototype; only the visualisation leaves the 7x7 grid. This
         is the positive final chapter for thesis A.
  protograd ~= 2.5
      -> a FOURTH rejected mechanism. The information is in the backbone (IG proves it)
         but the prototype pathway cannot expose it at pixel resolution either. The
         negative result is then close to airtight.
  per_prototype << protograd
      -> the aggregate is carried by summing many prototypes and INDIVIDUAL prototypes
         are still unlocalised. That would undercut the case-based claim even if the
         aggregate looks good -- check this before claiming success.""")


if __name__ == "__main__":
    main()
