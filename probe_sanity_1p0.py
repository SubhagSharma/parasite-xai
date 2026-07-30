"""
probe_sanity_1p0.py — WHY is ProtoPNet sanity_check still 1.0 after the patch?

The near-constant guard only fires if the randomised-model explanation is actually
near-constant by our threshold. This reproduces MPRT's top-down + skip_layers
randomisation on the REAL model and prints, for the scored (fully randomised) step:

  - std/|mean| of the randomised-model attribution per sample  (is it flat?)
  - the raw Spearman MPRT computes between trained and randomised attributions
  - what our _near_constant_mask decides at the current threshold

If std/|mean| is well below 1e-2 but the mask didn't fire in the eval, it's an
alignment bug. If std/|mean| is ABOVE 1e-2, the map isn't flat — the 1.0 comes from
somewhere else (e.g. the prototypes never being randomised) and needs a different fix.

CPU-only, reads best.pt, touches nothing.

    python probe_sanity_1p0.py \
        --config configs/generated/A2_protopnet_mobilevit.yaml \
        --ckpt   runs/A2_protopnet_mobilevit/best.pt
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from pxai.utils import load_config
from pxai.data import build_loaders
from pxai.models import build_model


def attr(model, x, target):
    ev = model.explain(x)
    sel = ev["proto_class"][:, target].t().view(target.size(0), -1, 1, 1)
    a = (ev["sim_maps"] * sel).sum(1, keepdim=True)
    return F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)


def rel_spread(a):
    f = a.reshape(a.shape[0], -1).detach().cpu().numpy().astype(np.float64)
    return f.std(1) / np.maximum(np.abs(f.mean(1)), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
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

    with torch.no_grad():
        a_trained = attr(model, x, y)
    rel_t = rel_spread(a_trained)

    # Replicate Quantus MPRT top-down + skip_layers: randomise every layer with
    # reset_parameters, accumulating, and score the FULLY randomised model.
    import copy
    rnd = copy.deepcopy(model)
    layers = [m for _, m in rnd.named_modules() if hasattr(m, "reset_parameters")]
    torch.manual_seed(0)
    for m in layers:
        m.reset_parameters()
    with torch.no_grad():
        a_rand = attr(rnd, x, y)
    rel_r = rel_spread(a_rand)

    # per-sample Spearman between trained and randomised (what MPRT correlates)
    at = a_trained.reshape(8, -1).numpy()
    ar = a_rand.reshape(8, -1).numpy()
    sp = np.array([stats.spearmanr(at[i], ar[i]).correlation for i in range(8)])

    print("\n=== per-sample diagnosis at the fully-randomised step ===")
    print(f"{'i':>3} {'rel_trained':>12} {'rel_random':>12} {'spearman(t,r)':>14}")
    for i in range(8):
        print(f"{i:>3} {rel_t[i]:>12.5f} {rel_r[i]:>12.5f} {sp[i]:>14.4f}")

    THRESH = 1e-2
    flagged = rel_r < THRESH
    print(f"\n  threshold _NEAR_CONSTANT_RELTOL = {THRESH}")
    print(f"  samples the guard WOULD flag (rel_random < thresh): {flagged.sum()}/8")
    print(f"  mean rel_random = {rel_r.mean():.5f}")
    print(f"  mean spearman(trained, randomised) = {np.nanmean(sp):.4f}")

    if rel_r.mean() < THRESH:
        print("\n  -> randomised map IS flat but guard didn't fire in eval => ALIGNMENT bug")
    elif np.nanmean(sp) > 0.9:
        print("\n  -> randomised map is NOT flat yet correlation ~1.0 => prototypes/features")
        print("     barely change under randomisation. Different fix needed (see notes).")
    else:
        print("\n  -> correlation is actually low here; the eval's 1.0 may be a stale results.json")


if __name__ == "__main__":
    main()