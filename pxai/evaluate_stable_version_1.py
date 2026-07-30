"""Full evaluation sweep — produces the three-axis result table + Pareto figure.

    python -m pxai.evaluate --config configs/default.yaml --ckpt runs/exp001/best.pt
    python -m pxai.evaluate --config ... --ckpt ... --resume     # skip finished methods

Runs, for the trained interpretable model AND post-hoc baselines on a heavy
black box:
  - accuracy (H4)
  - faithfulness via Quantus (H1) -> normalised aggregate
  - cost: passes-per-explanation + params/FLOPs/latency (H2)
  - trust: temperature-scaled ECE + conformal sets + risk-coverage (H5)
and writes results.json + faithfulness_vs_cost.png + risk_coverage.png.

results.json is rewritten after EVERY method, so a crash or a kill costs you one
explainer, not the whole sweep. Use --resume to pick up where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import torch

from .utils import load_config, pick_device, ensure_dir
from .data import build_loaders
from .models import build_model
from .explainers.posthoc import explain_posthoc
from .eval.faithfulness import evaluate_faithfulness, normalised_aggregates
from .eval import cost as costmod
from .eval import conformal as conf
from .plots.pareto import plot_faithfulness_vs_cost, plot_risk_coverage
from .train import evaluate_acc


def ante_hoc_attr(kind):
    """Unify an inherently-interpretable model's evidence into a (B,1,H,W) attribution.

    NOTE: the returned fn explains `m`, the model it is HANDED — not a captured
    one. Quantus's sanity_check passes progressively randomised deepcopies here;
    explaining a closed-over original instead makes the explanation invariant to
    randomisation, which scores as a perfect sanity-check failure (corr = 1.0).
    """
    import torch.nn.functional as F

    def fn(m, x, target):
        ev = m.explain(x)
        if kind == "protopnet":
            maps = ev["sim_maps"]                                   # (B,P,h,w)
            pc = ev["proto_class"]                                  # (P,C)
            sel = pc[:, target].t().view(target.size(0), -1, 1, 1)  # (B,P,1,1)
            a = (maps * sel).sum(1, keepdim=True)
        elif kind == "bcos":
            cm = ev["contrib_map"]                                  # (B,C,h,w)
            a = cm.gather(1, target.view(-1, 1, 1, 1).expand(-1, 1, *cm.shape[-2:]))
        else:  # cbm has no spatial map -> fall back to gradcam-style on features
            a = m.features(x).mean(1, keepdim=True)
        return F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return fn


def _empty_cache(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write(results, path):
    """Atomic-ish incremental dump so a kill never leaves a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)


def run(cfg, ckpt, resume: bool = False):
    device = pick_device(cfg["device"])
    loaders = build_loaders(cfg)
    cfg["model"]["num_classes"] = len(loaders.classes)
    out = ensure_dir(cfg["output_dir"])
    results_path = f"{out}/results.json"

    model = build_model(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    kind = cfg["model"]["kind"]

    ev_cfg = cfg.get("eval", {})
    fa_params = ev_cfg.get("faithfulness_params", {})
    fa_batches = ev_cfg.get("faithfulness_batches", 8)
    fa_metric_batches = ev_cfg.get("faithfulness_metric_batches", {})
    fa_metrics = ev_cfg["faithfulness"]

    results = {}
    if resume and os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"[evaluate] resuming; already have: "
              f"{sorted(results.get('methods', {}))}", flush=True)

    results.setdefault("methods", {})
    if "accuracy" not in results:
        results["accuracy"] = evaluate_acc(model, loaders.test, device)
        results["cost_model"] = costmod.cost_report(model, cfg["data"]["img_size"])
        results["eval_setup"] = {
            "batch_size": cfg["data"]["batch_size"],
            "img_size": cfg["data"]["img_size"],
            "faithfulness_batches": fa_batches,
            "images_scored": fa_batches * cfg["data"]["batch_size"],
            "metrics": fa_metrics,
            "params": fa_params,
        }
        _write(results, results_path)

    def _score(label, attr, passes):
        if label in results["methods"]:
            print(f"[evaluate] skip {label} (already done)", flush=True)
            return
        print(f"[evaluate] === {label} ===", flush=True)
        fa = evaluate_faithfulness(model, loaders.test, attr, fa_metrics, device,
                                   max_batches=fa_batches,
                                   metric_batches=fa_metric_batches,
                                   params=fa_params)
        _empty_cache(device)
        results["methods"][label] = {
            "faithfulness": fa["scores"],
            "n_samples": fa["n_samples"],
            "failures": fa["failures"],
            "batches_used": fa["batches_used"],
            "explainer_determinism": fa["explainer_determinism"],
            "sanity_collapse_batches": fa.get("sanity_collapse_batches", 0),
            "passes": passes,
        }
        _write(results, results_path)   # durable the moment a method finishes

    # --- ante-hoc (our model) ---
    if kind != "blackbox":
        _score(f"ours:{kind}", ante_hoc_attr(kind), costmod.EXPLAINER_PASSES[kind])

    # --- post-hoc baselines on the SAME model (swap to a heavy black box in practice) ---
    for name in cfg["explain"]["posthoc"]:
        attr = lambda m, x, t, _n=name: explain_posthoc(_n, m, x, t)[0]
        _score(name, attr, costmod.EXPLAINER_PASSES[name])

    # --- aggregate ACROSS methods (needs them all; cannot be done per-method) ---
    per_method = {k: v["faithfulness"] for k, v in results["methods"].items()}
    aggs = normalised_aggregates(per_method)
    rows = []
    for label, agg in aggs.items():
        results["methods"][label]["aggregate"] = agg
        rows.append({"method": label,
                     "passes": results["methods"][label]["passes"],
                     "faithfulness": agg,
                     "kind": "ante" if label.startswith("ours:") else "posthoc"})
    _write(results, results_path)

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
    _write(results, results_path)

    plot_faithfulness_vs_cost(rows, f"{out}/faithfulness_vs_cost.png")
    plot_risk_coverage({"DSS": (cov, risk)}, f"{out}/risk_coverage.png")
    print(json.dumps(results, indent=2))
    print(f"\nFigures + results.json written to {out}/")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="runs/exp001/best.pt")
    ap.add_argument("--resume", action="store_true",
                    help="skip methods already present in results.json")
    args = ap.parse_args()
    run(load_config(args.config), args.ckpt, resume=args.resume)
