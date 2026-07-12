"""Faithfulness evaluation harness (protocol §4a).

Wraps Quantus so every explanation — post-hoc, prototype, CBM, B-cos, amortized —
is scored on the same metrics. We report PER-METRIC scores AND a normalised
aggregate, because faithfulness metrics demonstrably disagree (the "disagreement
problem", arXiv:2311.07763 / 2404.11330) — a single number would cherry-pick.

Metrics:
  deletion / insertion  — Quantus RegionPerturbation / PixelFlipping AUC
  infidelity            — Quantus Infidelity (Yeh et al.)
  sensitivity           — Quantus MaxSensitivity (stability to input noise)
  sanity_check          — Adebayo model-randomisation (explanation MUST degrade)

`attr_fn(model, x, target) -> (B,1,H,W)` unifies all explainers.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
try:
    import quantus
except Exception:                                   # keep import-light for CI
    quantus = None


def _np(x):
    return x.detach().cpu().numpy()


def evaluate_faithfulness(model, loader, attr_fn, metrics, device, max_batches: int = 2):
    """Return {metric_name: mean_score}. attr_fn must return (B,1,H,W) attributions."""
    if quantus is None:
        raise ImportError("pip install quantus to run faithfulness evaluation")

    model.eval()
    metric_objs = _build_metrics(metrics)
    acc = {m: [] for m in metric_objs}

    for bi, (x, y) in enumerate(tqdm(loader, total=min(max_batches, len(loader)), desc="faithfulness batches")):
        if bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        a = attr_fn(model, x, y)                     # (B,1,H,W)
        a_np = _np(a)
        x_np, y_np = _np(x), _np(y)
        def _explain_func(model, inputs, targets, **kwargs):
            x_t = torch.as_tensor(inputs, device=device, dtype=torch.float32)
            y_t = torch.as_tensor(targets, device=device, dtype=torch.long)
            a = attr_fn(model, x_t, y_t)
            return _np(a)

        for name, metric in tqdm(metric_objs.items(), desc=f"batch {bi} metrics", leave=False):
            try:
                if name == "sensitivity":
                    # MaxSensitivity recomputes explanations on perturbed inputs itself,
                    # so it needs explain_func rather than a fixed a_batch.
                    scores = metric(model=model, x_batch=x_np, y_batch=y_np,
                                    a_batch=a_np, device=str(device),
                                    explain_func=_explain_func)
                else:
                    scores = metric(model=model, x_batch=x_np, y_batch=y_np,
                                    a_batch=a_np, device=str(device))
                acc[name].extend(np.asarray(scores).ravel().tolist())
            except Exception as e:
                acc[name].append(float("nan"))
                print(f"[faithfulness] {name} failed on batch {bi}: {e}")
    return {m: float(np.nanmean(v)) if v else float("nan") for m, v in acc.items()}


def _build_metrics(names):
    objs = {}
    if "deletion" in names:
        objs["deletion"] = quantus.RegionPerturbation(
            patch_size=14, order="morf", regions_evaluation=10, normalise=True)
    if "insertion" in names:
        objs["insertion"] = quantus.PixelFlipping(features_in_step=224, normalise=True)
    if "infidelity" in names:
        objs["infidelity"] = quantus.Infidelity(perturb_baseline="black", n_perturb_samples=10)
    if "sensitivity" in names:
        objs["sensitivity"] = quantus.MaxSensitivity(nr_samples=10, lower_bound=0.1)
    if "sanity_check" in names:
        objs["sanity_check"] = quantus.ModelParameterRandomisation(
            layer_order="independent", normalise=True)
    return objs


def normalised_aggregate(scores: dict) -> float:
    """Min-max-free aggregate: mean of metrics after sign-aligning (higher=better)."""
    higher_better = {"insertion", "sensitivity"}      # sensitivity: lower better -> invert
    vals = []
    for k, v in scores.items():
        if np.isnan(v):
            continue
        vals.append(-v if k in ("sensitivity", "infidelity", "deletion") else v)
    return float(np.mean(vals)) if vals else float("nan")
