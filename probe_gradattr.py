"""
probe_gradattr.py — does the gradient trick generalise beyond ProtoPNet?

BACKGROUND
----------
Every ante-hoc head in this project explains a prediction with a spatial map read off
the backbone's feature grid. On MobileViT-XS that grid is 7x7 on a 224px input, so one
cell is 32x32 px = 2.0% of the frame. Measured localisation of those native maps
(conc_pos; 1.0 = random, 16.3 = perfect on chula_roi2_w477):

    ours:protopnet  3.08      ours:cbm  1.43      ours:bcos  2.02
    integrated_gradients on the same models: 5.2-7.1

Three interventions failed to close that gap for ProtoPNet -- a corrected attribution
(1.08 -> 2.53), egg-constrained prototype push (3.08 -> 2.45, worse), and a 448px input
giving a 14x14 grid (2.53 -> 1.61, worse). All three stayed INSIDE the feature grid.

A fourth worked. Taking the gradient of each prototype's pooled similarity with respect
to the INPUT, at pixel resolution, gave conc_pos 5.46 and c@1% 11.80 -- matching IG
(5.31 / 12.40) while remaining a per-prototype decomposition. Five seeds, 275 images.

THIS PROBE ASKS THREE THINGS
----------------------------
1. DOES IT GENERALISE? Every head has an "interpretable intermediate quantity". The
   gradient trick should apply to all of them:

       protopnet   sim_pooled[p]        55 prototypes
       cbm         c[k] = sigmoid(...)  23 concepts
       bcos        class logit          no components (B-cos claims logit = W(x).x
                                        exactly, so grad x input IS the contribution)
       blackbox    class logit          no components -- THE CONTROL

2. IS IT JUST "GRADIENTS BEAT COARSE MAPS"? blackbox has no interpretable intermediate,
   so its gradattr is plain grad x input. If blackbox gains as much as the ante-hoc
   heads, the finding is about gradients generally, not about interpretability. What
   would remain specific to ante-hoc heads is the PER-COMPONENT decomposition -- being
   able to say "this region matched prototype 3" or "this region drove concept
   has_polar_plugs" -- which a blackbox saliency map cannot do at all.

3. HOW DOES IT COMPARE TO POST-HOC? Same table, same images, same metric.

VARIANTS
    native            the head's own spatial map (current ante-hoc attribution)
    gradattr          |d intermediate / dx| . x, weighted and summed   <- the proposal
    gradattr_smooth   SmoothGrad over 8 copies, sigma 0.10
    per_component     the single highest-weighted prototype / concept alone.
                      For bcos and blackbox there are no components, so this equals
                      gradattr and is reported as such.
    + any post-hoc method passed via --methods

DATASETS
Pass any runs; the dataset and its labels.json are read from each config. Note that
crops (box ~55% of frame) SATURATE this metric -- nothing can concentrate much above
1.0 -- so crop rows are for completeness, not for ranking. See report SEC 6.3.

    # generality: all four heads, one dataset
    python -u probe_gradattr.py --device cuda \\
        --runs roi477_protopnet_120ep,roi477_cbm_sup_120ep,roi477_bcos_120ep,roi477_blackbox_120ep

    # + post-hoc comparison (slower: lime and kernelshap are ~4x)
    python -u probe_gradattr.py --device cuda --runs ... \\
        --methods gradcam,hirescam,integrated_gradients,lime,kernelshap

    # across dataset versions
    python -u probe_gradattr.py --device cuda \\
        --runs roi477_protopnet_120ep,roi679_blackbox_120ep,crop_protopnet_120ep,A2_protopnet_mobilevit_120ep
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

HAS_COMPONENTS = {"protopnet", "protopnet_ms", "cbm"}


def labels_for(root):
    local = os.path.join(root, "labels.json")
    if os.path.exists(local):
        return local
    up = os.path.join(os.path.dirname(root.rstrip("/")),
                      "Chula-ParasiteEgg-11", "labels.json")
    return up if os.path.exists(up) else None


# ------------------------------------------------------- interpretable intermediates
def intermediate(model, kind, x, target):
    """-> (values (B,K), weights (B,K)).

    `values` are the head's interpretable intermediate quantities; `weights` are how
    much each contributes to the target class logit. The gradient of
    (values * weights).sum() wrt x is then the per-component attribution, and summing
    over K gives the class-level one.
    """
    head = getattr(model, "head", None)

    if kind == "protopnet":
        sp, _ = head._similarities(model.backbone(x))          # (B,P)
        w = head.last.weight
        if getattr(head, "pip_sparsity", False):
            w = F.relu(w)
        return sp, w[target]

    if kind == "protopnet_ms":
        # the multi-scale head takes the whole feature pyramid, not the last map
        from pxai.models.protopnet_multiscale import pyramid_forward
        sp, _ = head._similarities(pyramid_forward(model.backbone, x))
        w = head.last.weight
        if getattr(head, "pip_sparsity", False):
            w = F.relu(w)
        return sp, w[target]

    if kind == "cbm":
        feat = model.backbone(x)
        c_log = head.concept(head.pool(feat).flatten(1))       # (B,K)
        c = torch.sigmoid(c_log)
        # d logit[y] / d c_log[k] = W_l[y,k] * c[k] * (1-c[k])
        w = head.classifier.weight[target] * c * (1.0 - c)     # (B,K)
        return c_log, w.detach()

    # bcos and blackbox: the class logit itself, one "component"
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    return out.gather(1, target.view(-1, 1)), torch.ones(x.shape[0], 1, device=x.device)


def native_attr(model, kind, x, target):
    """The head's own spatial map, upsampled -- the current ante-hoc attribution."""
    head = getattr(model, "head", None)
    with torch.no_grad():
        feat = model.backbone(x)
        if kind == "protopnet_ms":
            from pxai.models.protopnet_multiscale import pyramid_forward
            feats = pyramid_forward(model.backbone, x)
            _, per_stage = head._similarities(feats)
            B, P = feat.shape[0], head.P
            W = head.last.weight
            if getattr(head, "pip_sparsity", False):
                W = F.relu(W)
            wy = W[target]                                     # (B,P)
            # upsample each stage's argmax-sparse map to input size, then sum
            a = torch.zeros(B, 1, *x.shape[-2:], device=x.device)
            for si, sim in enumerate(per_stage):
                idx = (head.stage_of == si).nonzero(as_tuple=True)[0]
                k, h, w2 = sim.shape[1], sim.shape[2], sim.shape[3]
                flat = sim.reshape(B, k, h * w2)
                mx, mi = flat.max(-1)
                sp = torch.zeros_like(flat)
                sp.scatter_(2, mi.unsqueeze(-1),
                            (mx * wy[:, idx]).unsqueeze(-1))
                a = a + F.interpolate(sp.view(B, k, h, w2).sum(1, keepdim=True),
                                      size=x.shape[-2:], mode="bilinear",
                                      align_corners=False)
            return a
        if kind == "protopnet":
            _, sim = head._similarities(feat)
            B, P, h, w = sim.shape
            W = head.last.weight
            if getattr(head, "pip_sparsity", False):
                W = F.relu(W)
            wy = W[target].view(B, P, 1, 1)
            flat = sim.reshape(B, P, h * w)
            mx, idx = flat.max(-1)
            sp = torch.zeros_like(flat)
            sp.scatter_(2, idx.unsqueeze(-1), (mx * wy.view(B, P)).unsqueeze(-1))
            a = sp.view(B, P, h, w).sum(1, keepdim=True)
        elif kind == "cbm":
            B, D = feat.shape[0], feat.shape[1]
            c = torch.sigmoid(head.concept(head.pool(feat).flatten(1)))
            wy = head.classifier.weight[target] * c * (1.0 - c)
            g = wy @ head.concept.weight
            a = (feat * g.view(B, D, 1, 1)).sum(1, keepdim=True)
        elif kind == "bcos":
            cm = head.block(feat)                              # (B,C,h,w)
            a = cm.gather(1, target.view(-1, 1, 1, 1).expand(-1, 1, *cm.shape[-2:]))
        else:
            return None
    return F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)


def grad_attr(model, kind, x, target, top_only=False, noise=0.0, n=1):
    """|d (weighted intermediate) / dx| . x, at pixel resolution."""
    total = torch.zeros_like(x[:, :1])
    for _ in range(max(1, n)):
        xi = x if noise <= 0 else x + torch.randn_like(x) * noise
        xi = xi.clone().detach().requires_grad_(True)
        vals, w = intermediate(model, kind, xi, target)
        if top_only and w.shape[1] > 1:
            k = int(w[0].argmax())
            obj = (vals[:, k] * w[:, k]).sum()
        else:
            obj = (vals * w).sum()
        g, = torch.autograd.grad(obj, xi)
        total = total + (g * xi).abs().sum(1, keepdim=True).detach()
    return total / max(1, n)


def metrics(a, mask):
    a = np.asarray(a, dtype=np.float64).ravel()
    m = np.asarray(mask, bool).ravel()
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
    ap.add_argument("--runs", required=True)
    ap.add_argument("--methods", default="", help="post-hoc methods to include")
    ap.add_argument("--n-per-class", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--emit-tsv", default="figs/gradattr.tsv")
    a = ap.parse_args()

    if a.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    posthoc = [m for m in a.methods.split(",") if m]
    if posthoc:
        from pxai.explainers.posthoc import explain_posthoc

    os.makedirs(os.path.dirname(a.emit_tsv) or ".", exist_ok=True)
    new = not os.path.exists(a.emit_tsv)
    tsv = open(a.emit_tsv, "a")
    if new:
        tsv.write("run\thead\tdataset\tclass\timage\tvariant\tarea\t"
                  "mass\tc1pct\tpeak\tfirst_hit_pct\n")

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

        ds = loaders.test.dataset
        base = ds.dataset if isinstance(ds, Subset) else ds
        while isinstance(base, Subset):
            base = base.dataset
        idxs = list(ds.indices) if isinstance(ds, Subset) else range(len(base.samples))
        rng = np.random.default_rng(a.seed)
        ncomp = "components" if kind in HAS_COMPONENTS else "no components (control)"
        print(f"\n{run}  ({kind}, {os.path.basename(root)}, grid {S//32}x{S//32}, "
              f"{ncomp})", flush=True)

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

                maps = {}
                nat = native_attr(model, kind, x, t)
                if nat is not None:
                    maps["native"] = nat
                try:
                    maps["gradattr"] = grad_attr(model, kind, x, t)
                    maps["gradattr_smooth"] = grad_attr(model, kind, x, t,
                                                        noise=0.10, n=8)
                    maps["per_component"] = grad_attr(model, kind, x, t, top_only=True)
                except Exception as e:
                    print(f"    gradattr failed on {os.path.basename(path)}: "
                          f"{type(e).__name__}: {e}", flush=True)
                for pm in posthoc:
                    try:
                        with torch.enable_grad():
                            maps[pm] = explain_posthoc(pm, model, x, t)[0]
                    except Exception:
                        pass

                for v, mp in maps.items():
                    arr = mp.detach().float().cpu().numpy()
                    arr = arr[0, 0] if arr.ndim == 4 else arr.squeeze()
                    mass, c1, pk, fh = metrics(arr, box)
                    if mass != mass:
                        continue
                    tsv.write(f"{run}\t{kind}\t{os.path.basename(root)}\t{cname}\t"
                              f"{os.path.basename(path)}\t{v}\t{box.mean():.4f}\t"
                              f"{mass:.4f}\t{c1:.4f}\t{pk:.0f}\t{fh:.4f}\n")
                    agg[(run, kind, v)]["mass"].append(mass)
                    agg[(run, kind, v)]["c1"].append(c1)
                    agg[(run, kind, v)]["first"].append(fh)
            tsv.flush()

    tsv.close()
    print(f"\nTSV -> {a.emit_tsv}")

    order = ["native", "gradattr", "gradattr_smooth", "per_component"] + posthoc
    print(f"\n{'run':<34}{'variant':<21}{'mass':>7}{'c@1%':>8}{'1st hit':>9}{'n':>5}")
    print("-" * 84)
    for k in sorted(agg, key=lambda k: (k[0], order.index(k[2])
                                        if k[2] in order else 99)):
        d = agg[k]
        print(f"{k[0]:<34}{k[2]:<21}{st.mean(d['mass']):>7.2f}{st.mean(d['c1']):>8.2f}"
              f"{st.mean(d['first']):>8.1f}%{len(d['mass']):>5}")

    # by head, pooled over runs -- the generality question
    byhead = collections.defaultdict(lambda: collections.defaultdict(list))
    for (run, kind, v), d in agg.items():
        for m in ("mass", "c1", "first"):
            byhead[(kind, v)][m] += d[m]
    print(f"\nBY HEAD (pooled over runs)\n{'head':<11}{'variant':<21}{'mass':>7}"
          f"{'c@1%':>8}{'1st hit':>9}{'gain':>7}{'n':>5}")
    print("-" * 70)
    for kind in ("protopnet", "cbm", "bcos", "blackbox"):
        nat = byhead.get((kind, "native"), {}).get("c1")
        for v in order:
            if (kind, v) not in byhead:
                continue
            d = byhead[(kind, v)]
            g = (st.mean(d["c1"]) / st.mean(nat)) if nat else float("nan")
            tag = f"{g:>6.1f}x" if g == g and v != "native" else "     -"
            print(f"{kind:<11}{v:<21}{st.mean(d['mass']):>7.2f}{st.mean(d['c1']):>8.2f}"
                  f"{st.mean(d['first']):>8.1f}%{tag}{len(d['mass']):>5}")
        print()

    print("""READING
  gain > 2x for protopnet, cbm AND bcos
      -> the failure is GENERAL to ante-hoc spatial maps and the fix generalises.
         The thesis claim becomes about ante-hoc visualisation, not about ProtoPNet.
  blackbox gains as much as the ante-hoc heads
      -> the mechanism is "gradients beat coarse feature maps", not anything about
         interpretability. What stays specific to ante-hoc heads is the PER-COMPONENT
         decomposition: only they can say WHICH prototype or concept a region drove.
         Report the control honestly and make the per-component claim the contribution.
  per_component ~= gradattr (protopnet, cbm)
      -> individual components localise as well as the sum, so "this looks like
         prototype 3" survives at the level of a single prototype.
  crops rows
      -> box is ~55% of frame, the metric saturates, nothing exceeds ~1.4. Do not rank
         crop numbers against the others (report SEC 6.3).""")


if __name__ == "__main__":
    main()
