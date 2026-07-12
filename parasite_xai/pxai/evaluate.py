"""Full evaluation sweep — produces the three-axis result table + Pareto figure.

    python -m pxai.evaluate --config configs/default.yaml --ckpt runs/exp001/best.pt

Runs, for the trained interpretable model AND post-hoc baselines on a heavy
black box:
  - accuracy (H4)
  - faithfulness via Quantus (H1) -> normalised aggregate
  - cost: passes-per-explanation + params/FLOPs/latency (H2)
  - trust: temperature-scaled ECE + conformal sets + risk-coverage (H5)
and writes results.json + faithfulness_vs_cost.png + risk_coverage.png.
"""
from __future__ import annotations

import argparse
import json
import torch

from .utils import load_config, pick_device, ensure_dir
from .data import build_loaders
from .models import build_model
from .explainers.posthoc import explain_posthoc
from .eval.faithfulness import evaluate_faithfulness, normalised_aggregate
from .eval import cost as costmod
from .eval import conformal as conf
from .plots.pareto import plot_faithfulness_vs_cost, plot_risk_coverage
from .train import evaluate_acc


def ante_hoc_attr(model, kind):
    """Unify an inherently-interpretable model's evidence into a (B,1,H,W) attribution."""
    import torch.nn.functional as F

    def fn(m, x, target):
        ev = model.explain(x)
        if kind == "protopnet":
            maps = ev["sim_maps"]                                   # (B,P,h,w)
            pc = ev["proto_class"]                                  # (P,C)
            sel = pc[:, target].t().view(target.size(0), -1, 1, 1)  # (B,P,1,1)
            a = (maps * sel).sum(1, keepdim=True)
        elif kind == "bcos":
            cm = ev["contrib_map"]                                  # (B,C,h,w)
            a = cm.gather(1, target.view(-1, 1, 1, 1).expand(-1, 1, *cm.shape[-2:]))
        else:  # cbm has no spatial map -> fall back to gradcam-style on features
            a = model.features(x).mean(1, keepdim=True)
        return F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return fn


def run(cfg, ckpt):
    device = pick_device(cfg["device"])
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    out = ensure_dir(cfg["output_dir"])

    model = build_model(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    kind = cfg["model"]["kind"]

    results = {"accuracy": evaluate_acc(model, loaders.test, device),
               "cost_model": costmod.cost_report(model, cfg["data"]["img_size"]),
               "methods": {}}

    rows = []  # for the Pareto figure

    # --- ante-hoc (our model) ---
    if kind != "blackbox":
        fa = evaluate_faithfulness(model, loaders.test, ante_hoc_attr(model, kind),
                                   cfg["eval"]["faithfulness"], device)
        agg = normalised_aggregate(fa)
        results["methods"][f"ours:{kind}"] = {"faithfulness": fa, "aggregate": agg,
                                              "passes": costmod.EXPLAINER_PASSES[kind]}
        rows.append({"method": kind, "passes": costmod.EXPLAINER_PASSES[kind],
                     "faithfulness": agg, "kind": "ante"})

    # --- post-hoc baselines on the SAME model (swap to a heavy black box in practice) ---
    for name in cfg["explain"]["posthoc"]:
        attr = lambda m, x, t, _n=name: explain_posthoc(_n, m, x, t)[0]
        fa = evaluate_faithfulness(model, loaders.test, attr,
                                   cfg["eval"]["faithfulness"], device)
        agg = normalised_aggregate(fa)
        results["methods"][name] = {"faithfulness": fa, "aggregate": agg,
                                    "passes": costmod.EXPLAINER_PASSES[name]}
        rows.append({"method": name, "passes": costmod.EXPLAINER_PASSES[name],
                     "faithfulness": agg, "kind": "posthoc"})

    # --- trust layer ---
    cal_logits, cal_y = conf.collect_logits(model, loaders.val, device)
    test_logits, test_y = conf.collect_logits(model, loaders.test, device)
    T = conf.fit_temperature(cal_logits, cal_y)
    probs = (test_logits / T).softmax(1)
    qhat = conf.conformal_qhat((cal_logits / T).softmax(1), cal_y, cfg["trust"]["alpha"])
    sets = conf.prediction_sets(probs, qhat)
    cov, risk = conf.risk_coverage(probs, test_y)
    results["trust"] = {
        "temperature": T,
        "ece": conf.expected_calibration_error(probs, test_y),
        "conformal_qhat": qhat,
        "avg_set_size": float(sets.sum(1).float().mean()),
        "empirical_coverage": float(sets.gather(1, test_y.view(-1, 1)).float().mean()),
    }

    plot_faithfulness_vs_cost(rows, f"{out}/faithfulness_vs_cost.png")
    plot_risk_coverage({"DSS": (cov, risk)}, f"{out}/risk_coverage.png")
    with open(f"{out}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nFigures + results.json written to {out}/")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="runs/exp001/best.pt")
    args = ap.parse_args()
    run(load_config(args.config), args.ckpt)
