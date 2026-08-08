"""
probe_concept_diagnose.py — why does `operculum` localise BETTER when it is absent,
and is the concept bottleneck causal or decorative?

TWO QUESTIONS, ONE RUN
======================

--- 1. THE NEGATIVE-GAP CONCEPTS -------------------------------------------------
probe_concept_parts measured, per concept, the attention concentration when the concept
is TRUE for the species minus when it is FALSE. Thirteen of 23 are positive as expected.
Ten are negative, and one is extreme:

    has_polar_plugs   2/11 classes   present 9.77   absent 4.65   gap +5.12
    operculum         3/11 classes   present 0.67   absent 5.72   gap -5.06

`operculum` at 0.67 when present is BELOW uniform: the slot attends away from the egg
precisely when the feature exists. That is not weak localisation, it is inverted, and it
is the largest single effect in the table.

HYPOTHESIS: the sign of the gap tracks which direction carries the discriminative
signal. A concept true in 3 of 11 classes is mostly a NEGATIVE test -- eight species out
of eleven are identified partly by *not* having an operculum. If the slot learns to
check the location where an operculum would be and report its absence, it will attend
sharply on the eight negative classes and diffusely on the three positive ones.

If that holds, this is not a defect. It is counterfactual localisation, and it changes
the claim from "the map shows the feature" to "the map shows WHERE THE FEATURE IS
DECIDED" -- which is more defensible and is a distinct thing to report.

TEST: correlate each concept's gap against its positive rate. A strong negative
correlation supports the hypothesis; no correlation refutes it and the inversion needs a
different explanation.

--- 2. IS THE BOTTLENECK CAUSAL? -------------------------------------------------
The concept head reaches 0.9882 macro balanced accuracy on 23 DPDx concepts. But
`c = M[y]` is a deterministic function of the class, so a model that classifies well can
emit the right concept vector WITHOUT the concepts driving the prediction. Accuracy
alone cannot distinguish a working bottleneck from a decorative one.

The standard test (Koh et al., 2020) is INTERVENTION: overwrite a concept at test time
and see whether the class prediction follows. `ConceptPartHead.forward` already accepts
`concept_intervene`.

    flip a DIAGNOSTIC concept  -> prediction should change often. The bottleneck is
                                  causal and the explanation is load-bearing.
    prediction unchanged       -> the class is being read from something the concepts do
                                  not mediate, and the bottleneck is decorative. That
                                  would be a serious limitation and must be reported.

A CONTROL IS INCLUDED. Flipping a concept the species does not have should matter LESS
than flipping one it does. Without that control, a high flip rate could just mean the
model is unstable under any perturbation.

    python -u probe_concept_diagnose.py --run roi477_parts_120ep --device cuda
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics as st

import numpy as np
import torch
from torch.utils.data import Subset

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.concepts_loader import load_concept_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="roi477_parts_120ep")
    ap.add_argument("--n", type=int, default=12, help="images per class")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-tsv", default="figs/concept_parts_eval.tsv",
                    help="output of probe_concept_parts, for question 1")
    a = ap.parse_args()

    cfg = load_config(f"configs/generated/{a.run}.yaml")
    cfg["device"] = a.device
    dev = pick_device(cfg["device"])
    loaders = build_loaders(cfg)
    classes = loaders.classes
    cfg["model"]["num_classes"] = len(classes)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(f"runs/{a.run}/best.pt", map_location=dev)["model"])
    model.eval()
    head = model.head
    table, names = load_concept_table(
        cfg["model"]["concept_parts"]["concepts_csv"], classes)
    table = table.to(dev)
    K = len(names)
    npos = table.sum(0).cpu().numpy()

    # ================================================== 1. why the negative gaps
    print("=== 1. DOES THE GAP SIGN TRACK CLASS CONTRAST? ===")
    gaps = {}
    if os.path.exists(a.eval_tsv):
        import csv
        pos = collections.defaultdict(list)
        neg = collections.defaultdict(list)
        for r in csv.DictReader(open(a.eval_tsv), delimiter="\t"):
            try:
                v = float(r["attn_conc"])
            except (ValueError, KeyError):
                continue
            if v != v:
                continue
            (pos if r["true"] == "1" else neg)[r["concept"]].append(v)
        for c in set(pos) | set(neg):
            if pos[c] and neg[c]:
                gaps[c] = st.mean(pos[c]) - st.mean(neg[c])
    if not gaps:
        print(f"  {a.eval_tsv} missing -- run probe_concept_parts.py first")
    else:
        xs, ys, lab = [], [], []
        for k, nm in enumerate(names):
            if nm in gaps:
                xs.append(float(npos[k]))
                ys.append(gaps[nm])
                lab.append(nm)
        r = float(np.corrcoef(xs, ys)[0, 1])
        print(f"  Pearson(positive class count, gap) = {r:+.3f}  n={len(xs)}")
        lo = [y for x, y in zip(xs, ys) if x <= 3]
        hi = [y for x, y in zip(xs, ys) if x >= 5]
        if lo and hi:
            print(f"  concepts in <=3 classes: mean gap {st.mean(lo):+.2f}  n={len(lo)}")
            print(f"  concepts in >=5 classes: mean gap {st.mean(hi):+.2f}  n={len(hi)}")
        print("""
  strong NEGATIVE correlation -> rare concepts localise WORSE when present, i.e. the
      slot is doing a negative test: it checks where the feature would be and reports
      its absence. Counterfactual localisation, not a defect -- but it changes the
      claim to "the map shows where the feature is DECIDED".
  no correlation -> the inversion has some other cause and needs separate diagnosis.""")

    # ================================================ 2. is the bottleneck causal?
    print("\n=== 2. CONCEPT INTERVENTION: is the bottleneck causal or decorative? ===")
    ds = loaders.test.dataset
    base = ds.dataset if isinstance(ds, Subset) else ds
    while isinstance(base, Subset):
        base = base.dataset
    idxs = list(ds.indices) if isinstance(ds, Subset) else list(range(len(base.samples)))
    rng = np.random.default_rng(a.seed)

    flip_true = collections.defaultdict(list)     # concept TRUE for species -> flipped
    flip_false = collections.defaultdict(list)    # concept FALSE -> flipped (CONTROL)
    n_img = 0

    for ci in range(len(classes)):
        pool = [i for i in idxs if base.samples[i][1] == ci]
        if not pool:
            continue
        for gi in [pool[j] for j in rng.choice(len(pool), min(a.n, len(pool)),
                                               replace=False)]:
            x, y = base[gi]
            x = x.unsqueeze(0).to(dev)
            with torch.no_grad():
                feat = model.features(x)
                base_logit, c_logit = head(feat)
                base_pred = int(base_logit.argmax(1))
                c = torch.sigmoid(c_logit)
            if base_pred != ci:
                continue                       # only intervene on correct predictions
            n_img += 1
            for k in range(K):
                iv = torch.full((1, K), float("nan"), device=dev)
                truth = int(table[ci, k])
                iv[0, k] = 0.0 if truth else 1.0        # flip it
                with torch.no_grad():
                    lg = head(feat, concept_intervene=iv)[0]
                changed = int(int(lg.argmax(1)) != base_pred)
                (flip_true if truth else flip_false)[k].append(changed)

    print(f"  {n_img} correctly-classified images, {K} concepts each\n")
    print(f"  {'concept':<32}{'classes+':>9}{'flip TRUE':>11}{'flip FALSE':>12}")
    rows = []
    for k in range(K):
        t = st.mean(flip_true[k]) if flip_true[k] else float("nan")
        f = st.mean(flip_false[k]) if flip_false[k] else float("nan")
        rows.append((k, t, f))
    for k, t, f in sorted(rows, key=lambda r: -(r[1] if r[1] == r[1] else -9)):
        print(f"  {names[k][:30]:<32}{int(npos[k]):>6}/11{t:>10.0%}{f:>11.0%}")

    tt = [r[1] for r in rows if r[1] == r[1]]
    ff = [r[2] for r in rows if r[2] == r[2]]
    if tt and ff:
        print(f"\n  mean flip rate, concept TRUE  : {st.mean(tt):.0%}")
        print(f"  mean flip rate, concept FALSE : {st.mean(ff):.0%}   (control)")
        print(f"  ratio {st.mean(tt) / max(st.mean(ff), 1e-9):.2f}x")
    print("""
  TRUE flip rate HIGH and above FALSE -> the bottleneck is CAUSAL. Removing a feature
      the species has changes the prediction; removing one it lacks matters less. The
      concept explanation is load-bearing, which accuracy alone could never show.
  both LOW -> the class is decided by something the concepts do not mediate. The
      bottleneck is decorative and this must be reported as a limitation.
  both HIGH and equal -> the model is simply unstable under any concept perturbation,
      and neither reading is supported.""")


if __name__ == "__main__":
    main()
