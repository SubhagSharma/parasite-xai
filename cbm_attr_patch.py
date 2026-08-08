"""
cbm_attr_patch.py — the correct CBM attribution, derived from the actual head.

THE BUG
-------
pxai/evaluate.py:56-57

    else:  # cbm has no spatial map -> fall back to gradcam-style on features
        a = m.features(x).mean(1, keepdim=True)

Two fatal properties:
  * `target` is never used -> the same map for every class. Not a class explanation.
  * the CBM head is never touched -> the concept bottleneck, the entire reason CBM is
    interpretable, is bypassed.

It is also not "gradcam-style": Grad-CAM is a gradient-weighted channel sum, this is
an unweighted mean. Every `ours:cbm` number in results.json therefore measures a
class-agnostic mean of backbone activations.

THE DERIVATION
--------------
From pxai/models/cbm.py, with hard_bottleneck=True:

    v      = GAP(feat)                     (B, D)
    c_log  = W_c v + b_c                   (B, K)      K = num_concepts
    c      = sigmoid(c_log)                (B, K)
    logit  = W_l c + b_l                   (B, C)

Everything is linear except an elementwise sigmoid on the concepts, so the gradient
of the target logit w.r.t. the feature map is available in closed form:

    d logit[y] / d feat[d,i,j]
        = (1/HW) * sum_k W_l[y,k] * c[k](1-c[k]) * W_c[k,d]

Define a per-sample channel weight

    g[d] = sum_k W_l[y,k] * c[k](1-c[k]) * W_c[k,d]

then Gradient x Input at the feature level is

    A[i,j] = sum_d g[d] * feat[d,i,j]

Class-dependent, concept-mediated, passes through the bottleneck, and EXACT — no
autograd, no approximation. Because it is closed-form it also works unchanged inside
`torch.no_grad()` and on the randomised deepcopies Quantus hands to MPRT.

TWO VARIANTS
------------
  mode="gradient"     weights concepts by W_l[y,k] * c[k](1-c[k])   <- default
  mode="contribution" weights concepts by W_l[y,k] alone

`gradient` is the true first-order sensitivity, which is what deletion and insertion
actually probe (remove a pixel, watch the logit move). `contribution` matches the
semantics of the existing `class_concept_contrib` in CBMHead.explain(), which is
W_l * c with no derivative term. They differ only by the c(1-c) factor. Report which
one you used; do not mix them.

CAVEAT — READ BEFORE USING THE NUMBERS
--------------------------------------
pxai/train.py:81 calls concept_loss(c_logit, None), which returns zero. The 16
concepts are UNSUPERVISED latent dimensions, not the named morphological concepts
(operculum, shell texture, polar knob) the docstring describes. This patch makes the
attribution correctly reflect what the model computes; it does not make the concepts
mean anything. CBM interpretability claims still require wiring up concepts.csv.

TO APPLY
--------
Replace the `else:` branch of ante_hoc_attr in pxai/evaluate.py with a call to
cbm_spatial_attr (see the snippet at the bottom of this file), or import it:

    from cbm_attr_patch import cbm_spatial_attr

Then re-run the CBM faithfulness eval. The ProtoPNet and B-cos branches are correct
and unaffected.

SELF-TEST
    python -u cbm_attr_patch.py --config configs/generated/roi477_cbm_120ep.yaml \
        --ckpt runs/roi477_cbm_120ep/best.pt
verifies the closed form against autograd and against the old attribution.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cbm_spatial_attr(m, x, target, mode: str = "gradient"):
    """(B,1,H,W) class-conditional attribution for an InterpretableModel with a CBMHead.

    Closed form, so it is safe under torch.no_grad() and on randomised copies.
    """
    head = m.head
    feat = m.backbone(x)                                  # (B, D, h, w)
    B, D, h, w = feat.shape

    v = head.pool(feat).flatten(1)                         # (B, D)
    c_log = head.concept(v)                               # (B, K)
    c = torch.sigmoid(c_log)                              # (B, K)

    Wl = head.classifier.weight                           # (C, K)
    Wc = head.concept.weight                              # (K, D)

    wy = Wl[target]                                       # (B, K)
    if mode == "gradient":
        wy = wy * c * (1.0 - c)                           # chain through the sigmoid
    elif mode != "contribution":
        raise ValueError(f"mode must be 'gradient' or 'contribution', got {mode!r}")

    g = wy @ Wc                                           # (B, D)
    a = (feat * g.view(B, D, 1, 1)).sum(1, keepdim=True)  # (B, 1, h, w)
    return F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- self-test
def _main():
    import argparse
    import numpy as np
    from scipy import stats

    from pxai.utils import load_config
    from pxai.data import build_loaders
    from pxai.models import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["device"] = "cpu"
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    m = build_model(cfg)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    m.eval()

    x, y = next(iter(loaders.test))
    x, y = x[:a.n], y[:a.n]
    C = len(loaders.classes)

    # 1. closed form vs autograd
    feat = m.backbone(x).detach().requires_grad_(True)
    logit, _ = m.head(feat)
    logit.gather(1, y.view(-1, 1)).sum().backward()
    auto = (feat.grad * feat.detach()).sum(1, keepdim=True) * feat.shape[-1] * feat.shape[-2]
    with torch.no_grad():
        mine = cbm_spatial_attr(m, x, y, "gradient")
    auto_up = F.interpolate(auto, size=x.shape[-2:], mode="bilinear", align_corners=False)
    rho = np.mean([stats.spearmanr(auto_up[i].flatten().numpy(),
                                   mine[i].flatten().numpy()).correlation
                   for i in range(a.n)])
    print(f"closed form vs autograd Grad x Input : spearman {rho:.6f}  "
          f"(1.000000 = exact)")

    # 2. is it class-conditional? the old attribution was not
    with torch.no_grad():
        alt = (y + 1) % C
        m_alt = cbm_spatial_attr(m, x, alt, "gradient")
        old = F.interpolate(m.features(x).mean(1, keepdim=True),
                            size=x.shape[-2:], mode="bilinear", align_corners=False)
    r_new = np.mean([stats.spearmanr(mine[i].flatten().numpy(),
                                     m_alt[i].flatten().numpy()).correlation
                     for i in range(a.n)])
    print(f"target y vs target y+1, NEW attribution: spearman {r_new:+.4f}  "
          f"(low = class-conditional)")
    print(f"target y vs target y+1, OLD attribution: spearman  1.0000  "
          f"(identical by construction — target is unused)")

    r_old = np.mean([stats.spearmanr(mine[i].flatten().numpy(),
                                     old[i].flatten().numpy()).correlation
                     for i in range(a.n)])
    print(f"NEW vs OLD attribution                 : spearman {r_old:+.4f}")

    # 3. gradient vs contribution mode
    with torch.no_grad():
        con = cbm_spatial_attr(m, x, y, "contribution")
    r_mode = np.mean([stats.spearmanr(mine[i].flatten().numpy(),
                                      con[i].flatten().numpy()).correlation
                      for i in range(a.n)])
    print(f"gradient mode vs contribution mode     : spearman {r_mode:+.4f}")

    print("\n--- patch for pxai/evaluate.py, ante_hoc_attr ---")
    print("""    else:  # cbm — exact class-conditional attribution through the bottleneck
        head = m.head
        feat = m.backbone(x)
        B, D = feat.shape[0], feat.shape[1]
        c = torch.sigmoid(head.concept(head.pool(feat).flatten(1)))
        wy = head.classifier.weight[target] * c * (1.0 - c)
        g = wy @ head.concept.weight
        a = (feat * g.view(B, D, 1, 1)).sum(1, keepdim=True)""")


if __name__ == "__main__":
    _main()
