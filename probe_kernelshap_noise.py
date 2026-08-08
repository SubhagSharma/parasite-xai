"""probe_kernelshap_noise.py — what is the tau gate actually measuring?

    python -u probe_kernelshap_noise.py --device cuda
    python -u probe_kernelshap_noise.py --runs roi477_bcos_120ep --n-images 40

THE QUESTION
------------
C2's gate is "FastSHAP reaches Kendall tau >= 0.70 against KernelSHAP". That
only means something if KernelSHAP agrees with ITSELF above 0.70. If two
KernelSHAP runs at different seeds correlate at, say, 0.45, then 0.70 against
it is unreachable no matter how good the amortised explainer is -- and you
would spend a month debugging an explainer that is not broken.

This is sharpened by the eps table: KernelSHAP scores 0.013 / 0.042 / 0.055 /
0.022 across the four datasets, the only method never exceeding 0.06. Maps that
carry almost no localisation signal are exactly the maps whose seed-to-seed
agreement is worth checking before building anything on top of them.

WHAT IT MEASURES
----------------
For each image, each explainer is run twice under different torch seeds and the
two maps are compared by Kendall tau at SUPERPIXEL granularity (16x16 = 256
groups, the same partition posthoc.py feeds to captum). Pixel-level tau would
be inflated: the maps are piecewise constant within a superpixel, so 150k
pixels carry only 256 independent values.

integrated_gradients is the control. It is deterministic given a fixed input,
so its tau must come back at ~1.000. If it does not, the measurement itself is
broken and the other two numbers mean nothing.

READING IT
----------
  ig      ~1.000            measurement is sound
  lime     high             the surrogate fit is stable across draws
  kshap    >= 0.85          the 0.70 gate has headroom; C2 is measurable
  kshap    0.70 - 0.85      gate is near the noise floor; report the ceiling
                            alongside any FastSHAP tau, always
  kshap    < 0.70           THE GATE IS UNREACHABLE. No amortised explainer can
                            agree with KernelSHAP better than KernelSHAP agrees
                            with itself. Renegotiate the C2 criterion before
                            writing any more FastSHAP code.

COST
----
KernelSHAP is 1000 forward passes per image per seed. At the default 20 images
that is ~40k passes for kshap plus ~40k for lime; roughly 15-25 min on the MIG
slice for a mobilevit_xs run. Scale --n-images with care.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from pxai.data import build_loaders
from pxai.models import build_model
from pxai.explainers.posthoc import explain_posthoc, SUPERPIXEL_GRID

try:
    from scipy.stats import kendalltau
except ImportError:                                          # pragma: no cover
    kendalltau = None


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def to_superpixels(attr, grid=SUPERPIXEL_GRID):
    """(B,1,H,W) -> (B, grid*grid). The map is piecewise constant on this
    partition, so this is lossless and avoids tie-inflated tau."""
    if attr.dim() == 3:
        attr = attr.unsqueeze(1)
    return F.adaptive_avg_pool2d(attr.float(), (grid, grid)).flatten(1)


def tau_pair(a, b):
    if kendalltau is None:
        raise SystemExit("scipy is required: pip install scipy")
    out = []
    for i in range(a.shape[0]):
        t = kendalltau(a[i].cpu().numpy(), b[i].cpu().numpy()).correlation
        if t == t:
            out.append(float(t))
    return out


def run_one(run, cfgp, ckpt, args, dev):
    cfg = load_config(cfgp)
    loaders = build_loaders(cfg)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    model.eval()

    xs, ys, seen = [], [], 0
    for x, y in loaders.val:
        take = min(args.batch, args.n_images - seen)
        xs.append(x[:take]); ys.append(y[:take]); seen += take
        if seen >= args.n_images:
            break
    X = torch.cat(xs).to(dev); Y = torch.cat(ys).to(dev)
    print(f"  {run}: {X.shape[0]} images", flush=True)

    res = {}
    for name in args.methods.split(","):
        taus = []
        for i in range(0, X.shape[0], args.batch):
            xb, yb = X[i:i + args.batch], Y[i:i + args.batch]
            maps = []
            for seed in (args.seed_a, args.seed_b):
                torch.manual_seed(seed)
                np.random.seed(seed)
                a, _ = explain_posthoc(name, model, xb, yb)
                maps.append(to_superpixels(a.detach()))
            taus += tau_pair(maps[0], maps[1])
        m = float(np.mean(taus)); sd = float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0
        ci = 1.96 * sd / max(1, np.sqrt(len(taus)))
        res[name] = {"tau": m, "ci": float(ci), "n": len(taus)}
        print(f"    {name:<22} tau = {m:.3f} +/- {ci:.3f}   n={len(taus)}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="roi477_bcos_120ep")
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--methods",
                    default="integrated_gradients,lime,kernelshap")
    ap.add_argument("--seed-a", type=int, default=1337)
    ap.add_argument("--seed-b", type=int, default=2337)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="figs/kernelshap_noise.json")
    a = ap.parse_args()
    dev = torch.device(a.device)

    out = {}
    for run in a.runs.split(","):
        for cfgp in sorted(glob.glob(f"configs/generated/{run}.yaml")):
            name = os.path.basename(cfgp)[:-5]
            ckpt = f"runs/{name}/best.pt"
            if not os.path.exists(ckpt):
                print(f"  skip {name} (no checkpoint)"); continue
            try:
                out[name] = run_one(name, cfgp, ckpt, a, dev)
            except Exception as e:                            # noqa: BLE001
                print(f"  FAIL {name}: {type(e).__name__}: {e}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 62)
    print("REFERENCE-NOISE CEILING -- read before interpreting any C2 tau")
    print("=" * 62)
    for run, r in out.items():
        print(f"\n{run}")
        for m, v in r.items():
            print(f"  {m:<24} {v['tau']:.3f} +/- {v['ci']:.3f}")
        ig = r.get("integrated_gradients", {}).get("tau")
        ks = r.get("kernelshap", {}).get("tau")
        if ig is not None and ig < 0.95:
            print("  ** IG is deterministic and should be ~1.000. It is not, so "
                  "this\n     measurement is unsound -- fix it before reading the "
                  "kernelshap row.")
        elif ks is not None:
            if ks < 0.70:
                print(f"  ** KernelSHAP agrees with itself at {ks:.3f}, BELOW the "
                      "0.70 C2 gate.\n     The gate is unreachable by construction. "
                      "Renegotiate the criterion.")
            elif ks < 0.85:
                print(f"  ** Ceiling is {ks:.3f}; the 0.70 gate sits close to the "
                      "noise floor.\n     Report this number alongside any FastSHAP "
                      "tau you quote.")
            else:
                print(f"  ceiling {ks:.3f} -- the 0.70 gate has headroom.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()