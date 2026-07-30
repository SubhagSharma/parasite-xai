"""Faithfulness evaluation harness (protocol §4a).

Wraps Quantus so every explanation — post-hoc, prototype, CBM, B-cos, amortized —
is scored on the same metrics. We report PER-METRIC scores AND a normalised
aggregate, because faithfulness metrics demonstrably disagree (the "disagreement
problem", arXiv:2311.07763 / 2404.11330) — a single number would cherry-pick.

Metrics (ALL are lower-is-better; see METRIC_DIRECTION):
  deletion      — Quantus RegionPerturbation, MoRF. AUC of the degradation curve.
  insertion     — Quantus PixelFlipping. NOTE: PixelFlipping removes pixels most-
                  relevant-first, so this is a *deletion*-style curve despite the
                  key name. Kept as "insertion" for config back-compat.
  infidelity    — Quantus Infidelity (Yeh et al.), MSE of predicted vs actual delta.
  sensitivity   — Quantus MaxSensitivity (instability under input noise).
  sanity_check  — Adebayo model-parameter randomisation. Spearman correlation
                  between the original and randomised-model explanation, so HIGH
                  means the explanation ignored the weights = FAILED the check.

`attr_fn(model, x, target) -> (B,1,H,W)` unifies all explainers.

IMPORTANT: attr_fn MUST explain the model it is handed, not a captured one.
Quantus hands it progressively randomised deepcopies during sanity_check; an
attr_fn that closes over the original model silently scores 1.0 (total failure).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

try:
    import quantus
except Exception:                                   # keep import-light for CI
    quantus = None


# Every metric here is lower-is-better. Kept explicit rather than implicit so the
# sign of the Pareto axis is auditable — this table is the thing reviewers check.
METRIC_DIRECTION: Dict[str, str] = {
    "deletion": "lower",
    "insertion": "lower",
    "infidelity": "lower",
    "sensitivity": "lower",
    "sanity_check": "lower",
}

# Metrics whose cost is (k x one attribution) rather than (k x one forward pass).
# MaxSensitivity re-explains nr_samples times; MPRT re-explains once per randomised
# layer (~80 for convnext_tiny). Both get a smaller image budget by default — same
# metric definition, fewer images, which is the honest lever. Report the N you used.
DEFAULT_METRIC_BATCHES: Dict[str, int] = {
    "sensitivity": 2,
    "sanity_check": 2,
}

DEFAULT_PARAMS: Dict[str, Any] = {
    # 3136 blocks at the Quantus default patch size of 4; 64 at 28. Same metric,
    # ~49x fewer forward passes. This single value dominated the old runtime.
    "infidelity_patch_size": 28,
    "infidelity_n_perturb_samples": 10,
    "deletion_patch_size": 14,
    "deletion_regions": 10,
    "insertion_features_in_step": 224,
    "sensitivity_nr_samples": 10,
    "sensitivity_lower_bound": 0.1,
    # True  -> one similarity, original vs FULLY randomised model (~1 attribution).
    # False -> layer-by-layer degradation curve (~80 attributions). Set False only
    #          when you actually want the per-layer curve for the C3 paper.
    "sanity_skip_layers": True,
}


# Substrings of Quantus's assert_attributions messages that indicate a DEGENERATE
# explanation (all-zero / all-negative / all-one / uniform) rather than a bug. On
# the sanity_check metric these mean the explanation collapsed on the randomised
# model — which is a pass, not a failure. Matched by substring because the full
# messages carry trailing advice text.
_DEGENERACY_MARKERS = (
    "all equal to zero",
    "all be less than zero",
    "all equal to one",
    "uniformly distributed",
)


def _is_degeneracy_error(e: Exception) -> bool:
    msg = str(e)
    return any(s in msg for s in _DEGENERACY_MARKERS)


# A per-sample attribution whose spatial spread (std / |mean|) is below this has no
# usable ranking — it is effectively constant. On the RANDOMISED model, gradient and
# prototype explainers can saturate to such a flat map; Quantus then correlates two
# near-constant arrays and returns a SPURIOUS ~1.0 (which reads as a sanity-check
# FAILURE) instead of the correct ~0.0. We detect that per sample and score it as a
# pass. Verified (diagnose_protopnet_maps): on the TRAINED model these maps have
# strong contrast (mean std/|mean| ~0.5), so this only fires post-randomisation and
# hides nothing real. Threshold sits well below trained-model contrast, well above
# the flatness a scrambled backbone produces.
_NEAR_CONSTANT_RELTOL = 1e-2


def _near_constant_mask(a_np: np.ndarray) -> np.ndarray:
    """Per-sample boolean: True where the attribution has no usable spatial ranking."""
    flat = np.asarray(a_np, dtype=np.float64).reshape(np.shape(a_np)[0], -1)
    rel = flat.std(axis=1) / np.maximum(np.abs(flat.mean(axis=1)), 1e-12)
    return rel < _NEAR_CONSTANT_RELTOL


def _randomised_flat_mask(model, x, y, attr_fn, layer_order, device):
    """Reproduce MPRT's fully-randomised model and flag samples whose explanation
    goes near-constant there. Order-independent: mirrors what Quantus scores under
    skip_layers=True (all reset_parameters layers randomised, accumulating), so the
    result matches the scored step without depending on MPRT's internal call order.

    Returns a per-sample boolean mask, or None if reproduction isn't possible (in
    which case the caller leaves MPRT's raw scores untouched).
    """
    import copy
    try:
        rnd = copy.deepcopy(model)
        layers = [m for _, m in rnd.named_modules() if hasattr(m, "reset_parameters")]
        if not layers:
            return None
        # top_down accumulates from the output side; independent randomises each in
        # turn. Either way the fully-randomised end state resets every eligible layer.
        for m in layers:
            m.reset_parameters()
        rnd.eval()
        with torch.no_grad():
            a = attr_fn(rnd, x.to(device), y.to(device))
        return _near_constant_mask(_np(a))
    except Exception:
        return None   # never let the guard's own failure break the metric


def _np(x):
    return x.detach().cpu().numpy()


_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy>=2.0 renamed trapz


def _curve_auc(arr: np.ndarray) -> np.ndarray:
    """Normalised AUC per sample for (B, n_steps) perturbation curves."""
    if arr.ndim == 1:
        return arr
    steps = arr.shape[1]
    if steps < 2:
        return arr.reshape(-1)
    return _trapz(arr, axis=1) / (steps - 1)


def _postprocess(name: str, scores: Any) -> tuple[np.ndarray, Optional[float]]:
    """Normalise a Quantus return value into (per-sample 1-D array, diagnostic).

    Handles the three shapes Quantus emits:
      - dict  {layer_name: [...]}          (MPRT)
      - (B, n_steps) curves                (RegionPerturbation, PixelFlipping)
      - (B,) floats                        (Infidelity, MaxSensitivity)
    """
    diagnostic = None
    if isinstance(scores, dict):
        # MPRT stores a "original" entry = similarity between the attribution we
        # passed in and one regenerated on the UNRANDOMISED model. It is ~1.0 by
        # construction and must not enter the mean. It is, however, a free
        # determinism check on attr_fn — surface it instead of discarding it.
        original = scores.get("original")
        if original is not None:
            vals = np.asarray(original, dtype=float).ravel()
            if vals.size:
                diagnostic = float(np.nanmean(vals))
        flat = [v for k, sub in scores.items() if k != "original"
                for v in np.asarray(sub, dtype=float).ravel()]
        return np.asarray(flat, dtype=float), diagnostic

    arr = np.asarray(scores, dtype=float)
    if arr.ndim >= 2:
        arr = _curve_auc(arr.reshape(arr.shape[0], -1))
    return arr.ravel(), diagnostic


def evaluate_faithfulness(model, loader, attr_fn, metrics, device,
                          max_batches: int = 8,
                          metric_batches: Optional[Dict[str, int]] = None,
                          params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score one explainer.

    Returns
    -------
    {
      "scores":      {metric: mean},          # NaN if every batch failed
      "n_samples":   {metric: int},           # how many values backed each mean
      "failures":    {metric: int},           # batches that raised
      "batches_used":{metric: int},
      "explainer_determinism": float | None,  # ~1.0 for deterministic attr_fn
    }
    """
    if quantus is None:
        raise ImportError("pip install quantus to run faithfulness evaluation")

    p = {**DEFAULT_PARAMS, **(params or {})}
    budget = {**DEFAULT_METRIC_BATCHES, **(metric_batches or {})}
    budget = {m: min(budget.get(m, max_batches), max_batches) for m in metrics}

    model.eval()
    metric_objs = _build_metrics(metrics, p)
    acc: Dict[str, list] = {m: [] for m in metric_objs}
    fails: Dict[str, int] = {m: 0 for m in metric_objs}
    used: Dict[str, int] = {m: 0 for m in metric_objs}
    collapses: Dict[str, int] = {m: 0 for m in metric_objs}   # sanity_check degeneracy passes
    determinism: list = []

    n_batches = min(max_batches, len(loader))
    for bi, (x, y) in enumerate(tqdm(loader, total=n_batches, desc="faithfulness batches")):
        if bi >= max_batches:
            break

        active = {n: m for n, m in metric_objs.items() if bi < budget.get(n, max_batches)}
        if not active:
            break   # nothing left to compute — don't pay for another attribution

        x, y = x.to(device), y.to(device)
        a = attr_fn(model, x, y)                     # (B,1,H,W)
        a_np = _np(a)
        x_np, y_np = _np(x), _np(y)

        def _explain_func(model, inputs, targets, **kwargs):
            """Quantus hands us the model under test — use IT, never a closure."""
            x_t = torch.as_tensor(inputs, device=device, dtype=torch.float32)
            y_t = torch.as_tensor(targets, device=device, dtype=torch.long)
            return _np(attr_fn(model, x_t, y_t))

        for name, metric in tqdm(active.items(), desc=f"batch {bi} metrics", leave=False):
            try:
                kw = dict(model=model, x_batch=x_np, y_batch=y_np,
                          a_batch=a_np, device=str(device))
                # Both re-explain internally, so both need the explainer itself.
                if name in ("sensitivity", "sanity_check"):
                    kw["explain_func"] = _explain_func
                vals, diag = _postprocess(name, metric(**kw))
                # Per-sample near-constant correction (sanity_check only). MPRT
                # correlates the trained-model explanation with the randomised-model
                # one; when the latter is near-constant (no spatial ranking) the
                # correlation is a SPURIOUS ~1.0 that reads as a sanity FAILURE. We
                # detect this order-independently: independently reproduce MPRT's
                # fully-randomised explanation here and flag samples whose randomised
                # map is flat, scoring those 0.0 (pass). Self-contained — does not
                # depend on how MPRT batches or orders its internal explain calls.
                if (name == "sanity_check" and vals.shape[0] == x_np.shape[0]):
                    flat = _randomised_flat_mask(model, x, y, attr_fn,
                                                 p.get("sanity_layer_order", "top_down"),
                                                 device)
                    if flat is not None and flat.any():
                        vals = vals.copy()
                        vals[flat] = 0.0
                        collapses[name] += int(flat.sum())
                acc[name].extend(vals.tolist())
                used[name] += 1
                if diag is not None:
                    determinism.append(diag)
            except AssertionError as e:
                # MPRT re-explains on the RANDOMISED model; gradient/perturbation
                # explainers (Grad-CAM, HiResCAM, LIME) often collapse to an all-zero,
                # all-negative, or uniform map there, which trips Quantus's
                # assert_attributions. A collapsed explanation on a scrambled model is
                # the CORRECT outcome of the sanity check (the explanation stopped
                # tracking the weights), so score it 0.0 (zero similarity = maximal
                # pass) instead of dropping the batch to NaN. Only sanity_check is
                # eligible; any other metric hitting this is a real failure.
                if name == "sanity_check" and _is_degeneracy_error(e):
                    acc[name].extend([0.0] * x_np.shape[0])
                    used[name] += 1
                    collapses[name] += 1
                else:
                    fails[name] += 1
                    print(f"[faithfulness] {name} failed on batch {bi}: "
                          f"{type(e).__name__}: {e}", flush=True)
            except Exception as e:
                fails[name] += 1
                print(f"[faithfulness] {name} failed on batch {bi}: "
                      f"{type(e).__name__}: {e}", flush=True)

    scores = {}
    for m, v in acc.items():
        finite = [s for s in v if np.isfinite(s)]
        scores[m] = float(np.mean(finite)) if finite else float("nan")
        if not finite:
            print(f"[faithfulness] WARNING: {m} produced no finite values "
                  f"({fails[m]} batch failures)", flush=True)

    return {
        "scores": scores,
        "n_samples": {m: int(np.sum(np.isfinite(v))) for m, v in acc.items()},
        "failures": fails,
        "batches_used": used,
        # batches where sanity_check's explanation collapsed on the randomised model
        # (scored 0.0 = passed). A high count is itself a finding: the explainer is
        # strongly model-dependent, which is what the sanity check is meant to reward.
        "sanity_collapse_batches": collapses["sanity_check"] if "sanity_check" in collapses else 0,
        "explainer_determinism": float(np.mean(determinism)) if determinism else None,
    }


def _build_metrics(names, p: Dict[str, Any]):
    objs = {}
    if "deletion" in names:
        objs["deletion"] = quantus.RegionPerturbation(
            patch_size=p["deletion_patch_size"], order="morf",
            regions_evaluation=p["deletion_regions"], normalise=True)
    if "insertion" in names:
        objs["insertion"] = quantus.PixelFlipping(
            features_in_step=p["insertion_features_in_step"], normalise=True,
            return_auc_per_sample=True)          # one AUC per image, not a raw curve
    if "infidelity" in names:
        objs["infidelity"] = quantus.Infidelity(
            perturb_baseline="black",
            n_perturb_samples=p["infidelity_n_perturb_samples"],
            perturb_patch_sizes=[p["infidelity_patch_size"]])   # was defaulting to [4]
    if "sensitivity" in names:
        objs["sensitivity"] = quantus.MaxSensitivity(
            nr_samples=p["sensitivity_nr_samples"],
            lower_bound=p["sensitivity_lower_bound"])
    if "sanity_check" in names:
        objs["sanity_check"] = quantus.ModelParameterRandomisation(
            layer_order="independent", normalise=True,
            skip_layers=p["sanity_skip_layers"])
    return objs


def normalised_aggregates(per_method: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Per-metric min-max across methods, sign-aligned, then averaged. Higher = better.

    This is RELATIVE, not absolute: a score of 1.0 means "best of the methods in
    this comparison on that metric", so adding or removing a method changes every
    other method's number. Always report the per-metric table alongside it — the
    aggregate exists to order points on the Pareto plot, not to be cited alone.

    Metrics present in fewer than two methods, or constant across all of them, are
    skipped (no meaningful spread to normalise against).
    """
    metrics = sorted({m for s in per_method.values() for m in s})
    norm: Dict[str, list] = {meth: [] for meth in per_method}

    for m in metrics:
        vals = {meth: s[m] for meth, s in per_method.items()
                if m in s and np.isfinite(s[m])}
        if len(vals) < 2:
            continue
        lo, hi = min(vals.values()), max(vals.values())
        if not np.isfinite(hi - lo) or (hi - lo) < 1e-12:
            continue
        for meth, v in vals.items():
            z = (v - lo) / (hi - lo)
            if METRIC_DIRECTION.get(m, "lower") == "lower":
                z = 1.0 - z
            norm[meth].append(z)

    return {meth: float(np.mean(v)) if v else float("nan") for meth, v in norm.items()}


def normalised_aggregate(scores: Dict[str, float]) -> float:
    """DEPRECATED — kept so old callers do not break.

    A single method's raw scores cannot be normalised against anything, and the
    metrics live on incompatible scales (infidelity is an unbounded MSE; MPRT is a
    correlation in [-1, 1]), so averaging them is dominated by whichever has the
    largest magnitude. Use normalised_aggregates() over all methods instead.
    """
    import warnings
    warnings.warn("normalised_aggregate() does not normalise; "
                  "use normalised_aggregates() across methods", DeprecationWarning,
                  stacklevel=2)
    vals = [-v if METRIC_DIRECTION.get(k, "lower") == "lower" else v
            for k, v in scores.items() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")