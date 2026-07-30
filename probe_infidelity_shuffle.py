r"""
probe_infidelity_shuffle.py — does infidelity measure SPATIAL faithfulness at all?

Scale-invariance of the optimal-scaled variant is proven algebraically. What is NOT
established is whether either variant can tell a real explanation from a fake one
with the same statistics. This is the standard control:

    for each explainer, compare its real attribution against a SPATIALLY SHUFFLED
    copy — identical values, identical histogram, identical magnitude and density,
    but the values are in the wrong places.

A metric that measures spatial faithfulness MUST score the real map better than its
own shuffle. If it cannot, the metric carries no spatial information and should not
appear in the paper as a faithfulness measure, however well-behaved its scaling is.

Reports both variants side by side so you can report whichever survives.

    python probe_infidelity_shuffle.py \
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
from pxai.eval.faithfulness import _build_metrics, DEFAULT_PARAMS


@torch.no_grad()
def ante_attr(m, x, target):
    ev = m.explain(x)
    sel = ev["proto_class"][:, target].t().view(target.size(0), -1, 1, 1)
    a = (ev["sim_maps"] * sel).sum(1, keepdim=True)
    return F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)


def shuffled(a_np, seed=0):
    """Same values, same histogram, same magnitude+density — wrong places."""
    out = a_np.reshape(a_np.shape[0], -1).copy()
    for i in range(out.shape[0]):
        np.random.RandomState(seed + i).shuffle(out[i])
    return out.reshape(a_np.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=16, help="images")
    ap.add_argument("--device", default="cpu", help="cpu | cuda  (cuda is ~20x faster)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dev = args.device
    cfg["device"] = dev
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    model = build_model(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.to(dev).eval()

    x, y = next(iter(loaders.test))
    x, y = x[: args.n].to(dev), y[: args.n].to(dev)
    xn, yn = x.cpu().numpy(), y.cpu().numpy()

    p = {**DEFAULT_PARAMS, **cfg.get("eval", {}).get("faithfulness_params", {})}
    objs = _build_metrics(["infidelity", "infidelity_scaled"], p)

    methods = {"ours:protopnet": None}
    for n in cfg["explain"]["posthoc"]:
        methods[n] = n

    print(f"\n=== shuffle control  ({len(y)} images) ===")
    print("A metric with spatial information scores REAL below SHUFFLED.\n")
    print(f"{'method':>22} {'variant':>10} {'real':>13} {'shuffled':>13} {'verdict':>12}")

    summary = {"infidelity": [0, 0], "infidelity_scaled": [0, 0]}
    for name, ph in methods.items():
        try:
            a = ante_attr(model, x, y) if ph is None else explain_posthoc(ph, model, x, y)[0]
        except Exception as e:
            print(f"{name:>22}  FAILED: {type(e).__name__}: {str(e)[:50]}")
            continue
        a_np = a.detach().cpu().numpy().astype(np.float32)
        a_sh = shuffled(a_np)

        for mname in ("infidelity", "infidelity_scaled"):
            try:
                r = float(np.mean(objs[mname](model=model, x_batch=xn, y_batch=yn,
                                              a_batch=a_np, device=dev)))
                s = float(np.mean(objs[mname](model=model, x_batch=xn, y_batch=yn,
                                              a_batch=a_sh, device=dev)))
            except Exception as e:
                print(f"{name:>22} {mname[-6:]:>10}  FAILED: {str(e)[:40]}")
                continue
            ok = r < s
            summary[mname][0 if ok else 1] += 1
            tag = "real wins" if ok else "SHUFFLE WINS"
            short = "legacy" if mname == "infidelity" else "scaled"
            print(f"{name:>22} {short:>10} {r:>13.4e} {s:>13.4e} {tag:>12}")

    print("\n=== verdict ===")
    for mname, (win, lose) in summary.items():
        tot = win + lose
        if tot == 0:
            continue
        short = "legacy" if mname == "infidelity" else "optimal-scaled"
        print(f"  {short:>16}: real beat shuffle {win}/{tot}")
        if win == tot:
            print(f"      -> carries spatial information; safe to report.")
        elif win == 0:
            print(f"      -> NO spatial information. Do not report as faithfulness.")
        else:
            print(f"      -> inconsistent; treat as unreliable and report per-method.")
    print()


if __name__ == "__main__":
    main()