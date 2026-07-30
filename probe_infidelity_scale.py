r"""
probe_infidelity_scale.py — why is ProtoPNet's infidelity 1.96e12?

Quantus Infidelity computes, per random 28x28 patch:

    loss = ( (f(x) - f(x_perturbed))  -  sum( attribution * removed_pixel_values ) )^2
             \________ pred_delta ________/   \____________ a_sum ______________/

pred_delta is IDENTICAL across explainers (same model, same perturbation).
a_sum scales LINEARLY with the explainer's raw attribution magnitude, summed over
28*28*3 = 2352 elements (the (B,1,H,W) attribution is broadcast across channels).

So if one explainer's raw values are ~50x another's, its infidelity is ~2500x larger
with no difference in faithfulness. This script measures both terms directly so you
can see whether ProtoPNet is genuinely unfaithful or just numerically large.

Also reports what normalise_by_max would do -- note Quantus normalises over ALL axes
by default (one global max for the whole batch), not per sample.

CPU-only, reads best.pt, changes nothing.

    python probe_infidelity_scale.py \
        --config configs/generated/A2_protopnet_mobilevit_120ep.yaml \
        --ckpt   runs/A2_protopnet_mobilevit_120ep/best.pt
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from pxai.utils import load_config
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.explainers.posthoc import explain_posthoc


@torch.no_grad()
def ante_attr(m, x, target):
    ev = m.explain(x)
    sel = ev["proto_class"][:, target].t().view(target.size(0), -1, 1, 1)
    a = (ev["sim_maps"] * sel).sum(1, keepdim=True)
    return F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)


def normalise_by_max_global(a):
    """Replicate quantus.normalise_by_max default: max over ALL axes (incl. batch)."""
    a = np.asarray(a, dtype=np.float64)
    if np.all(a == 0.0):
        return a
    return a / np.max(np.abs(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--patch", type=int, default=28)
    ap.add_argument("--n-patches", type=int, default=8, help="random patches to sample")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["device"] = "cpu"
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    model = build_model(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()

    x, y = next(iter(loaders.test))
    x, y = x[:8], y[:8]
    B = x.shape[0]

    methods = {"ours:protopnet": None}
    for n in cfg["explain"]["posthoc"]:
        methods[n] = n

    # pick random patches once so every method sees the SAME perturbations
    rng = np.random.RandomState(0)
    H, W = x.shape[-2:]
    ps = args.patch
    patches = [(int(rng.randint(0, H - ps)), int(rng.randint(0, W - ps)))
               for _ in range(args.n_patches)]

    # pred_delta is method-independent -- compute once, under no_grad.
    # NOTE: no_grad must NOT wrap the attribution loop below; hires_cam and
    # integrated_gradients need a live autograd graph.
    with torch.no_grad():
        f_x = model(x)[torch.arange(B), y]
        deltas = []
        for (r, c) in patches:
            xp = x.clone()
            xp[:, :, r:r+ps, c:c+ps] = 0.0        # "black" baseline
            f_xp = model(xp)[torch.arange(B), y]
            deltas.append((f_x - f_xp).numpy())
        deltas = np.stack(deltas, 1)               # (B, n_patches)

    print(f"\n=== pred_delta (identical for every explainer) ===")
    print(f"  |logit change| mean = {np.abs(deltas).mean():.4f}   max = {np.abs(deltas).max():.4f}")

    print(f"\n=== per-explainer attribution scale and infidelity decomposition ===")
    print(f"{'method':>22} {'raw max':>11} {'raw mean':>11} {'a_sum RAW':>13} "
          f"{'a_sum NORM':>12} {'infid RAW':>12} {'infid NORM':>12}")

    rows = []
    for name, posthoc_name in methods.items():
        try:
            if posthoc_name is None:
                a = ante_attr(model, x, y)
            else:
                a = explain_posthoc(posthoc_name, model, x, y)[0]
        except Exception as e:
            print(f"{name:>22}  FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        a_np = a.detach().cpu().numpy()             # (B,1,H,W)

        a_norm = normalise_by_max_global(a_np)

        a_sums_raw, a_sums_norm = [], []
        for (r, c) in patches:
            # x_diff = original pixels inside the patch, 0 elsewhere
            xd = np.zeros_like(x.numpy())
            xd[:, :, r:r+ps, c:c+ps] = x.numpy()[:, :, r:r+ps, c:c+ps]
            # attribution broadcast across the 3 channels, exactly as Quantus does
            ab_raw = np.broadcast_to(a_np, x.shape)
            ab_norm = np.broadcast_to(a_norm, x.shape)
            a_sums_raw.append((ab_raw * xd).reshape(B, -1).sum(-1))
            a_sums_norm.append((ab_norm * xd).reshape(B, -1).sum(-1))
        a_sums_raw = np.stack(a_sums_raw, 1)
        a_sums_norm = np.stack(a_sums_norm, 1)

        infid_raw = ((deltas - a_sums_raw) ** 2).mean()
        infid_norm = ((deltas - a_sums_norm) ** 2).mean()

        print(f"{name:>22} {np.abs(a_np).max():>11.3f} {np.abs(a_np).mean():>11.4f} "
              f"{np.abs(a_sums_raw).mean():>13.2e} {np.abs(a_sums_norm).mean():>12.3f} "
              f"{infid_raw:>12.3e} {infid_norm:>12.3e}")
        rows.append((name, infid_raw, infid_norm))

    print("\n=== ranking under each treatment (lower = better) ===")
    for label, idx in (("RAW (no normalisation)", 1), ("NORMALISED", 2)):
        order = sorted(rows, key=lambda r: r[idx])
        print(f"  {label}:")
        for r in order:
            print(f"      {r[0]:>22} {r[idx]:.3e}")

    print("\n  If the RAW ranking matches your results.json, normalise is NOT being")
    print("  applied in the eval. If the NORMALISED ranking reorders the methods,")
    print("  the infidelity column in the paper is measuring attribution SCALE.\n")


if __name__ == "__main__":
    main()