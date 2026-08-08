"""
probe_protopnet_attr.py — is ProtoPNet's poor localisation the model, or the attribution?

THE PROBLEM
-----------
`ante_hoc_attr` in pxai/evaluate.py builds ProtoPNet's explanation as

    a = (sim_maps * proto_class[:, target]).sum(1)          # (B,1,h,w)

The forward pass computes something different, in two respects.

**1. Spatial reduction.** protopnet.py:87

    sim_pooled = F.max_pool2d(sim, sim.shape[-2:])          # ONE location per prototype
    logits     = F.linear(sim_pooled, W)

The logit depends only on the **argmax location of each prototype**. Every other
spatial position has *exactly zero* influence. The attribution instead sums the entire
dense field. Since sim = log((d+1)/(d+1e-4)) is bounded and non-zero everywhere — a
mediocre patch still scores a moderate positive value — those fields carry a high floor.
Summing five of them yields a smooth map dominated by *average* prototype proximity
rather than by the five peaks that decide the class.

**2. Weights.** `proto_class` is a fixed identity buffer registered at construction;
`self.last.weight` is initialised from it and then **learned**, and is ReLU'd when
`pip_sparsity` is set. The attribution uses the buffer, the forward pass uses the
learned weight.

WHY THIS MATTERS
----------------
Measured on roi477_protopnet_120ep, the current attribution gives conc_pos 1.08 and
c@1% 1.44 — uniform at every scale, ranked last of 23 explanations. If that is an
artefact of the two mismatches above, the finding is "the standard way of visualising
ProtoPNet does not localise" and there is a fix. If the corrected attribution scores the
same, the finding is "ProtoPNet's explanations do not localise" and it stands.

THE FOUR VARIANTS
-----------------
    field_protoclass  the current implementation. Baseline; should reproduce ~1.44.
    field_lastw       dense field, correct learned weights. Isolates mismatch 2.
    argmax_sparse     EXACT. Only each prototype's argmax location contributes, at
                      value W[y,p] * sim_max[p]. This is literally what the logit is a
                      sum of. Isolates mismatch 1 (and 2).
    argmax_soft       spatial softmax with temperature instead of a hard argmax — a
                      smooth surrogate that keeps the peak emphasis without discarding
                      the field entirely.

Reported per variant: conc_pos (mass), c@1% (ranking), peak-in-box, first-hit rank.

    python -u probe_protopnet_attr.py \
        --runs roi477_protopnet_120ep,roi477_protopnet_s2337_120ep --device cuda
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

VARIANTS = ("field_protoclass", "field_lastw", "argmax_sparse", "argmax_soft")


def class_weights(head, target):
    """(B,P) — the weights the FORWARD PASS applies to pooled similarities."""
    w = head.last.weight                                     # (C,P), learned
    if getattr(head, "pip_sparsity", False):
        w = F.relu(w)                                        # PIP-Net non-negativity
    return w[target]


def attributions(model, x, target):
    """-> dict of (1,1,H,W) maps, one per variant, all upsampled to input size."""
    head = model.head
    feat = model.backbone(x)
    _, sim = head._similarities(feat)                        # (B,P,h,w)
    B, P, h, w = sim.shape
    out = {}

    pc = head.proto_class[:, target].t().view(B, P, 1, 1)    # fixed buffer
    lw = class_weights(head, target).view(B, P, 1, 1)        # learned, ReLU'd

    out["field_protoclass"] = (sim * pc).sum(1, keepdim=True)
    out["field_lastw"] = (sim * lw).sum(1, keepdim=True)

    # EXACT: place W[y,p]*sim_max[p] at each prototype's argmax, zero elsewhere.
    flat = sim.reshape(B, P, h * w)
    mx, idx = flat.max(-1)                                   # (B,P)
    sparse = torch.zeros_like(flat)
    sparse.scatter_(2, idx.unsqueeze(-1), (mx * lw.view(B, P)).unsqueeze(-1))
    out["argmax_sparse"] = sparse.view(B, P, h, w).sum(1, keepdim=True)

    # Soft surrogate: spatial softmax keeps peak emphasis without discarding the field.
    sm = F.softmax(flat * 4.0, dim=-1).view(B, P, h, w)
    out["argmax_soft"] = (sim * sm * lw).sum(1, keepdim=True)

    return {k: F.interpolate(v, size=x.shape[-2:], mode="bilinear", align_corners=False)
            for k, v in out.items()}


def metrics(a, mask):
    """-> mass_conc, c@1%, peak, first_hit_pct."""
    a = np.asarray(a, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    n = a.size
    area = float(m.mean())
    if area <= 0 or not np.isfinite(a).all():
        return (float("nan"),) * 4
    pos = np.clip(a, 0, None)
    mass = float(pos[m].sum() / pos.sum() / area) if pos.sum() > 0 else float("nan")
    order = np.argsort(-a, kind="stable")
    inbox = m[order]
    t = max(1, int(round(0.01 * n)))
    c1 = float(inbox[:t].mean()) / area
    peak = float(inbox[0])
    hit = np.flatnonzero(inbox)
    first = float(hit[0] / n * 100.0) if hit.size else float("nan")
    return mass, c1, peak, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="roi477_protopnet_120ep")
    ap.add_argument("--n-per-class", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--emit-tsv", default="figs/protopnet_attr_variants.tsv")
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
        S, root = cfg["data"]["img_size"], cfg["data"]["root"]
        if cfg["model"]["kind"] != "protopnet":
            print(f"SKIP {run}: not a protopnet head")
            continue
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

        head = model.head
        print(f"\n{run}: {head.prototypes.shape[0]} prototypes, "
              f"pip_sparsity={getattr(head, 'pip_sparsity', False)}, "
              f"feature grid {S // 32}x{S // 32}")

        ds = loaders.test.dataset
        base = ds.dataset if isinstance(ds, Subset) else ds
        while isinstance(base, Subset):
            base = base.dataset
        idxs = list(ds.indices) if isinstance(ds, Subset) else range(len(base.samples))
        rng = np.random.default_rng(a.seed)

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
                with torch.no_grad():
                    maps = attributions(model, x, t)
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
    print(f"\n{'run':<32}{'variant':<19}{'mass':>7}{'c@1%':>7}{'peak%':>7}{'1st hit':>9}{'n':>5}")
    print("-" * 86)
    for k in sorted(agg, key=lambda k: (k[0], VARIANTS.index(k[1]))):
        v = agg[k]
        print(f"{k[0]:<32}{k[1]:<19}{st.mean(v['mass']):>7.2f}{st.mean(v['c1']):>7.2f}"
              f"{st.mean(v['peak'])*100:>6.0f}%{st.mean(v['first']):>8.1f}%{len(v['mass']):>5}")

    print("""
REFERENCE  a perfect explanation scores ~16.3 on this dataset (mean 1/box-area);
           1.0 is random. IG scores 12.4, the best post-hoc method 13.6.

READING
  argmax_sparse >> field_protoclass  -> the reported ProtoPNet failure was an
      ATTRIBUTION defect, not a model one. Fix ante_hoc_attr, re-run the faithfulness
      and localisation evaluations for every protopnet run, and rewrite SEC 6 and 7.4.
  argmax_sparse ~= field_protoclass  -> the model genuinely does not localise. The
      existing conclusion stands and is now properly evidenced against the exact
      attribution rather than a proxy for it.
  field_lastw vs field_protoclass    -> isolates how much the wrong weights alone cost.""")


if __name__ == "__main__":
    main()
