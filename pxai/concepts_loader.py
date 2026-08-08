"""Class-level morphology concepts for the CBM bottleneck.

Loads a species x concept CSV, one-hot encodes the categorical columns, and hands
back a (num_classes, num_concepts) target matrix aligned to the dataset's class
order, plus per-concept pos_weight for the BCE.

SCOPE — STATE THIS IN THE WRITEUP
---------------------------------
These are CLASS-LEVEL concepts: c = M[y] is a deterministic function of the label.
That is weaker than Koh et al. (2020), whose CUB annotations are PER IMAGE and carry
genuine within-class variation ("this bird's crown is occluded in this photo"). Here
there is none: every Ascaris image gets an identical concept target. A reviewer will
observe that the concepts are a re-parameterisation of the label, and they are right.

The measurement it still enables is real, and it is the reason to do this:

    class accuracy 0.998 with concept accuracy 0.85  =>  THE BOTTLENECK LEAKS.

A hard bottleneck routes the label through c only, so the model cannot be right about
the species while being wrong about the morphology -- unless information is reaching
the classifier by some route the concept vector does not describe. Measuring that gap
is a faithfulness result, and it does not depend on the concepts being per-image.

BALANCE
-------
12 of the 23 encoded concepts are positive for exactly one species out of eleven.
Under unweighted BCE a model scores ~0.91 on those by predicting negative always, so
raw per-concept accuracy is uninformative. Two consequences, both handled here or in
the reporting:
  * pos_weight = n_neg / n_pos per concept, so a rare positive carries proportionate
    gradient (Ascaris's mammillated shell gets weight 10.0, not 1.0);
  * report BALANCED accuracy per concept, i.e. mean(TPR, TNR). `raw` is kept beside
    it only to show the gap.
"""
from __future__ import annotations

import csv
from typing import List, Sequence, Tuple

import torch


def load_concept_table(csv_path: str, classes: Sequence[str],
                       key: str = "species") -> Tuple[torch.Tensor, List[str]]:
    """-> (C, K) float target matrix in `classes` order, and K concept names.

    Raises on any class missing from the CSV. A silent drop here would train the
    bottleneck against zeros for that species and look like a model failure.
    """
    rows = list(csv.DictReader(open(csv_path)))
    by_species = {r[key]: r for r in rows}

    missing = [c for c in classes if c not in by_species]
    if missing:
        raise KeyError(
            f"{len(missing)} dataset class(es) absent from {csv_path}: {missing}\n"
            f"CSV has: {sorted(by_species)}\n"
            f"Run validate_concepts.py --classes-from <dataset root> first.")

    cols = [c for c in rows[0] if c != key]
    schema = []                                   # (column, kind, values)
    for c in cols:
        vals = sorted({r[c] for r in rows})
        schema.append((c, "binary", vals) if set(vals) <= {"0", "1"}
                      else (c, "onehot", vals))

    names, mat = [], []
    for c, kind, vals in schema:
        names += [c] if kind == "binary" else [f"{c}={v}" for v in vals]
    for cls in classes:
        r = by_species[cls]
        row = []
        for c, kind, vals in schema:
            if kind == "binary":
                row.append(float(int(r[c])))
            else:
                row += [1.0 if r[c] == v else 0.0 for v in vals]
        mat.append(row)

    return torch.tensor(mat, dtype=torch.float32), names


def concept_pos_weight(table: torch.Tensor, cap: float = 20.0) -> torch.Tensor:
    """n_neg / n_pos per concept, capped. Concepts with no positive get weight 1."""
    pos = table.sum(0)
    neg = table.shape[0] - pos
    w = torch.where(pos > 0, neg / pos.clamp(min=1.0), torch.ones_like(pos))
    return w.clamp(max=cap)


def concept_targets(table: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B,) labels -> (B, K) concept targets. This is the c = M[y] map."""
    return table.to(y.device).index_select(0, y)


@torch.no_grad()
def concept_report(pred_logit: torch.Tensor, target: torch.Tensor,
                   names: Sequence[str], thresh: float = 0.5):
    """Per-concept balanced accuracy. Returns (rows, macro_balanced, macro_raw).

    Balanced = mean(TPR, TNR), so predicting the majority everywhere scores 0.5 no
    matter how skewed the concept is. Raw accuracy on a 1-of-11 concept scores 0.91
    for the same degenerate behaviour, which is why it must not be reported alone.
    """
    p = (torch.sigmoid(pred_logit) > thresh).float()
    rows = []
    for k, nm in enumerate(names):
        t, q = target[:, k], p[:, k]
        npos, nneg = t.sum().item(), (1 - t).sum().item()
        tpr = ((q == 1) & (t == 1)).sum().item() / npos if npos else float("nan")
        tnr = ((q == 0) & (t == 0)).sum().item() / nneg if nneg else float("nan")
        raw = (q == t).float().mean().item()
        bal = ((tpr + tnr) / 2 if npos and nneg
               else (tpr if npos else tnr))
        rows.append({"concept": nm, "balanced": bal, "raw": raw,
                     "tpr": tpr, "tnr": tnr, "n_pos": int(npos)})
    fin = [r["balanced"] for r in rows if r["balanced"] == r["balanced"]]
    return (rows,
            sum(fin) / len(fin) if fin else float("nan"),
            sum(r["raw"] for r in rows) / len(rows))


def print_concept_report(rows, macro_bal, macro_raw, class_acc=None):
    print(f"\n{'concept':<32}{'balanced':>10}{'raw':>8}{'TPR':>8}{'TNR':>8}{'n_pos':>8}")
    print("-" * 74)
    for r in sorted(rows, key=lambda r: r["balanced"]):
        print(f"  {r['concept']:<30}{r['balanced']:>10.4f}{r['raw']:>8.4f}"
              f"{r['tpr']:>8.3f}{r['tnr']:>8.3f}{r['n_pos']:>8}")
    print("-" * 74)
    print(f"  {'MACRO':<30}{macro_bal:>10.4f}{macro_raw:>8.4f}")
    if class_acc is not None:
        print(f"\n  class accuracy   {class_acc:.4f}")
        print(f"  concept balanced {macro_bal:.4f}")
        gap = class_acc - macro_bal
        print(f"  GAP              {gap:+.4f}")
        if gap > 0.10:
            print("\n  THE BOTTLENECK LEAKS. The model is getting the species right")
            print("  while getting the morphology wrong, so information is reaching the")
            print("  classifier by a route the concept vector does not describe. With a")
            print("  hard bottleneck that should be impossible -- report this gap, and")
            print("  check whether the 16->23 concept width is being used as a")
            print("  general-purpose code rather than as the named concepts.")
        elif gap < -0.05:
            print("\n  Concepts are predicted BETTER than the class. The bottleneck is")
            print("  not the limiting factor; the classifier head is.")
        else:
            print("\n  Concept and class accuracy agree: the bottleneck is doing the work")
            print("  it claims to do.")
