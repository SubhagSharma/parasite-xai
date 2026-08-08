"""
probe_sanity_v2.py — is sanity_check a MEASUREMENT or an artefact of the correction?

WHAT PROMPTED THIS
------------------
Night 5 faithfulness on chula_roi2_w477:

    ours:protopnet        sanity_check =  0.0000     <- exactly zero
    gradcam on protopnet  sanity_check =  0.0000     <- exactly zero, same value
    ours:cbm              sanity_check = -0.6668     <- strongly NEGATIVE
    gradcam on blackbox   sanity_check =  0.2122     <- looks like a real measurement
    hirescam on blackbox  sanity_check =  0.0115

Two different explainers returning identical 0.0 to full precision is not a
measurement. faithfulness.py has exactly two code paths that write a hard 0.0:

  PATH A  _randomised_flat_mask fires -> vals[flat] = 0.0     (per-sample)
  PATH B  AssertionError degeneracy   -> [0.0] * batch        (whole batch)

Both are deliberate and defensible in isolation: a collapsed explanation on a
scrambled model IS the correct sanity outcome. But if EVERY sample takes one of
those paths, MPRT's raw correlation was never used, and "sanity_check = 0.0" means
"the correction fired 100% of the time", not "the explanation passed the check".
Those are very different claims and only one belongs in a paper.

This probe reproduces both paths on the real checkpoint and reports which fired,
per sample, alongside the RAW MPRT correlation that would have been reported
without any correction.

THE SIGN PROBLEM (independent of the above)
-------------------------------------------
METRIC_DIRECTION maps sanity_check -> "lower". MPRT scores the Spearman correlation
between the trained-model and randomised-model explanations. What you want is
|rho| near zero: the explanation stopped tracking the weights. rho = -0.67 means the
explanation tracks the weights just as strongly as +0.67 does, merely inverted --
but under "lower is better" it is ranked BETTER than a genuine 0.0 pass.

So CBM's -0.6668 currently scores as the best sanity result in the table when it is
in fact a strong (inverted) weight dependence. This probe prints both rho and |rho|
so the effect on the ranking is visible. FIX: use |rho|, or map the direction of
sanity_check through an absolute value before normalised_aggregate() sees it.

GENERALISED OVER HEADS
----------------------
probe_sanity_1p0.py hard-codes ProtoPNet's evidence dict (`proto_class`, `sim_maps`)
and 8 samples. This version dispatches on model.explain()'s output for protopnet,
cbm and bcos, and falls back to input-gradient for blackbox, so all four heads and
the post-hoc baselines can be compared on one axis.

    python -u probe_sanity_v2.py \
        --config configs/generated/roi477_protopnet_120ep.yaml \
        --ckpt   runs/roi477_protopnet_120ep/best.pt --n 32
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from pxai.utils import load_config
from pxai.data import build_loaders
from pxai.models import build_model

NEAR_CONSTANT_RELTOL = 1e-2          # must match faithfulness.py


# --------------------------------------------------------------------- attribution
def make_attr(kind):
    """Return attr(model, x, y) -> (B,1,H,W), matching faithfulness.py's ante_hoc_attr.

    IMPORTANT: takes `model` as an argument rather than closing over it. The closure
    version of this was the bug that made sanity_check 1.0 for every interpretable
    head -- the randomised copy was built, then the attribution was computed against
    the ORIGINAL model anyway.
    """
    def upsample(a, x):
        return F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def protopnet(model, x, y):
        ev = model.explain(x)
        sel = ev["proto_class"][:, y].t().view(y.size(0), -1, 1, 1)
        return upsample((ev["sim_maps"] * sel).sum(1, keepdim=True), x)

    def cbm(model, x, y):
        ev = model.explain(x)
        for k in ("concept_maps", "concept_act", "maps"):
            if k in ev and ev[k].dim() == 4:
                w = ev.get("concept_class")
                if w is not None:
                    sel = w[:, y].t().view(y.size(0), -1, 1, 1)
                    return upsample((ev[k] * sel).sum(1, keepdim=True), x)
                return upsample(ev[k].mean(1, keepdim=True), x)
        raise KeyError(f"no 4-D concept map in explain(): {list(ev)}")

    def bcos(model, x, y):
        ev = model.explain(x)
        for k in ("contrib_map", "contribs", "maps"):
            if k in ev:
                a = ev[k]
                if a.dim() == 4 and a.shape[1] > 1:
                    a = a[torch.arange(y.size(0)), y].unsqueeze(1)
                return upsample(a, x)
        raise KeyError(f"no contribution map in explain(): {list(ev)}")

    def gradient(model, x, y):
        x = x.clone().requires_grad_(True)
        out = model(x)
        out.gather(1, y.view(-1, 1)).sum().backward()
        return x.grad.abs().sum(1, keepdim=True)

    return {"protopnet": protopnet, "cbm": cbm, "bcos": bcos}.get(kind, gradient)


def rel_spread(a):
    f = np.asarray(a, dtype=np.float64).reshape(np.shape(a)[0], -1)
    return f.std(1) / np.maximum(np.abs(f.mean(1)), 1e-12)


def randomise(model, seed=0):
    rnd = copy.deepcopy(model)
    layers = [m for _, m in rnd.named_modules() if hasattr(m, "reset_parameters")]
    torch.manual_seed(seed)
    for m in layers:
        m.reset_parameters()
    rnd.eval()
    return rnd, len(layers)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=3,
                    help="randomisation seeds; MPRT uses one, but a single seed can "
                         "collapse by luck. Disagreement across seeds is itself a result.")
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg["device"] = a.device
    dev = torch.device(a.device)
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
    model.eval()

    kind = cfg["model"]["kind"]
    attr = make_attr(kind)
    print(f"{cfg['backbone']['name']} + {kind}   n={a.n}   "
          f"reltol={NEAR_CONSTANT_RELTOL}")

    xs, ys = [], []
    for x, y in loaders.test:
        xs.append(x)
        ys.append(y)
        if sum(t.size(0) for t in xs) >= a.n:
            break
    x = torch.cat(xs)[:a.n].to(dev)
    y = torch.cat(ys)[:a.n].to(dev)

    try:
        with torch.enable_grad():
            a_t = attr(model, x, y).detach().cpu().numpy()
        trained_ok = True
    except Exception as e:
        print(f"  trained-model attribution FAILED: {type(e).__name__}: {e}")
        return
    rel_t = rel_spread(a_t)

    print(f"\n{'seed':>5}{'layers':>8}{'threw':>8}{'flat %':>9}"
          f"{'mean rho':>11}{'mean |rho|':>12}{'reported':>10}")
    print("-" * 63)

    summary = []
    for s in range(a.seeds):
        rnd, nlayers = randomise(model, seed=s)
        threw = ""
        try:
            with torch.enable_grad():
                a_r = attr(rnd, x, y).detach().cpu().numpy()
        except Exception as e:
            # PATH B in faithfulness.py: whole batch scored 0.0
            print(f"{s:>5}{nlayers:>8}{type(e).__name__[:7]:>8}"
                  f"{'-':>9}{'-':>11}{'-':>12}{0.0:>10.4f}")
            summary.append((s, None, None, None, "PATH B (exception -> 0.0)"))
            continue

        rel_r = rel_spread(a_r)
        flat = rel_r < NEAR_CONSTANT_RELTOL          # PATH A
        ft = a_t.reshape(a.n, -1)
        fr = a_r.reshape(a.n, -1)
        rho = np.array([stats.spearmanr(ft[i], fr[i]).correlation for i in range(a.n)])
        rho = np.nan_to_num(rho, nan=0.0)

        reported = rho.copy()
        reported[flat] = 0.0                          # what faithfulness.py writes
        print(f"{s:>5}{nlayers:>8}{threw or '-':>8}{flat.mean() * 100:>8.1f}%"
              f"{np.nanmean(rho):>11.4f}{np.nanmean(np.abs(rho)):>12.4f}"
              f"{reported.mean():>10.4f}")
        summary.append((s, flat, rho, reported, None))

    print("\n=== reading ===")
    live = [t for t in summary if t[1] is not None]
    if not live:
        print("  Every seed threw during attribution on the randomised model.")
        print("  faithfulness.py scores that batch 0.0 via PATH B. The reported 0.0")
        print("  is a DEGENERACY PASS, not a measured correlation. Report it as")
        print("  'explanation collapsed on the randomised model', with the collapse")
        print("  count, never as a numeric sanity score.")
        return

    fl = float(np.mean([t[1].mean() for t in live]))
    raw = float(np.mean([np.nanmean(t[2]) for t in live]))
    absr = float(np.mean([np.nanmean(np.abs(t[2])) for t in live]))
    rep = float(np.mean([t[3].mean() for t in live]))

    print(f"  samples corrected to 0.0 by the flat guard : {fl * 100:.1f}%")
    print(f"  RAW mean rho (no correction)               : {raw:+.4f}")
    print(f"  mean |rho|  (what the metric SHOULD use)   : {absr:.4f}")
    print(f"  value faithfulness.py reports              : {rep:+.4f}")

    if fl > 0.95:
        print("\n  ARTEFACT. The guard fires on essentially every sample, so the")
        print("  reported score is the correction, not MPRT. The explanation goes")
        print("  near-constant on the randomised model -- which IS a pass, but state")
        print("  it as a collapse rate, not as a correlation of 0.0.")
    elif fl > 0.05:
        print(f"\n  MIXED. {fl*100:.0f}% of samples are corrected and the rest carry a")
        print("  real correlation. The mean blends two different quantities. Report")
        print("  the collapse rate and the uncorrected mean separately.")
    else:
        print("\n  CLEAN. The guard barely fires; this is a genuine MPRT measurement.")

    if abs(raw) > 0.2 and absr - abs(raw) < 0.05:
        sign = "negative" if raw < 0 else "positive"
        print(f"\n  SIGN WARNING: raw rho is consistently {sign} ({raw:+.4f}).")
        print("  METRIC_DIRECTION has sanity_check = 'lower', so a strong NEGATIVE")
        print(f"  correlation scores BETTER than a true pass at 0.0. |rho| = {absr:.4f}")
        print("  is the honest number: the explanation still tracks the weights, just")
        print("  inverted. Take the absolute value before normalised_aggregate().")

    if len(live) > 1:
        spread = max(np.nanmean(t[2]) for t in live) - min(np.nanmean(t[2]) for t in live)
        if spread > 0.15:
            print(f"\n  SEED SENSITIVITY: rho varies by {spread:.3f} across "
                  f"{len(live)} seeds.")
            print("  MPRT uses one randomisation. Report a seed mean, not a point value.")


if __name__ == "__main__":
    main()
