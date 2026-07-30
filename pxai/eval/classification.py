"""Classification metrics (protocol 6.4, accuracy axis).

Top-1 accuracy is a weak summary for an 11-class long-tail problem: a model can
score well while failing a rare species entirely. The proposal specifies macro-F1
and per-species recall precisely because they are long-tail-aware -- every class
contributes equally regardless of support.

    from pxai.eval.classification import evaluate_classification
    m = evaluate_classification(model, loaders.test, device, class_names)

Returns accuracy, macro/weighted F1, macro recall (= balanced accuracy), per-class
precision/recall/F1/support, the confusion matrix, and macro AUROC when scores are
available. Pure numpy -- no sklearn dependency.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return (y_true, y_pred, probs) as numpy arrays."""
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = torch.softmax(logits.float(), dim=1)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _prf(cm: np.ndarray):
    """Per-class precision, recall, F1, support from a confusion matrix."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)          # true count per class
    pred_count = cm.sum(0).astype(float)       # predicted count per class
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def _auroc_ovr(y_true, probs) -> Optional[float]:
    """Macro one-vs-rest AUROC via the rank (Mann-Whitney) formulation."""
    n_classes = probs.shape[1]
    aucs = []
    for c in range(n_classes):
        pos = probs[y_true == c, c]
        neg = probs[y_true != c, c]
        if len(pos) == 0 or len(neg) == 0:
            continue                            # class absent -> undefined, skip
        scores = np.concatenate([pos, neg])
        order = scores.argsort()
        ranks = np.empty(len(scores), dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        # average ranks over ties so the statistic stays correct
        _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sums = np.zeros(len(counts))
            np.add.at(sums, inv, ranks)
            ranks = (sums / counts)[inv]
        r_pos = ranks[: len(pos)].sum()
        aucs.append((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return float(np.mean(aucs)) if aucs else None


def evaluate_classification(model, loader, device,
                            class_names: Optional[List[str]] = None) -> Dict:
    y_true, y_pred, probs = collect_predictions(model, loader, device)
    n_classes = probs.shape[1]
    names = class_names or [str(i) for i in range(n_classes)]
    cm = confusion_matrix(y_true, y_pred, n_classes)
    precision, recall, f1, support = _prf(cm)
    present = support > 0                       # ignore classes absent from this split
    w = support / max(support.sum(), 1)

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1[present].mean()) if present.any() else float("nan"),
        "weighted_f1": float((f1 * w).sum()),
        # macro recall == balanced accuracy; the long-tail-aware headline number
        "macro_recall": float(recall[present].mean()) if present.any() else float("nan"),
        "macro_precision": float(precision[present].mean()) if present.any() else float("nan"),
        "auroc_macro_ovr": _auroc_ovr(y_true, probs),
        "per_class": {
            names[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i]), "support": int(support[i])}
            for i in range(n_classes)
        },
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


def print_report(m: Dict, title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    print(f"  accuracy      {m['accuracy']:.4f}")
    print(f"  macro F1      {m['macro_f1']:.4f}")
    print(f"  macro recall  {m['macro_recall']:.4f}   (balanced accuracy)")
    print(f"  weighted F1   {m['weighted_f1']:.4f}")
    if m["auroc_macro_ovr"] is not None:
        print(f"  AUROC (OvR)   {m['auroc_macro_ovr']:.4f}")
    print(f"\n  {'class':>24} {'prec':>7} {'recall':>7} {'F1':>7} {'n':>6}")
    for name, d in sorted(m["per_class"].items(), key=lambda kv: kv[1]["f1"]):
        flag = "  <-- weakest" if d["f1"] == min(
            v["f1"] for v in m["per_class"].values() if v["support"] > 0) else ""
        print(f"  {name[:24]:>24} {d['precision']:>7.3f} {d['recall']:>7.3f} "
              f"{d['f1']:>7.3f} {d['support']:>6d}{flag}")
