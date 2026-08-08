#!/usr/bin/env python
# apply_concept_supervision.py -- wire concepts_v3.csv into the CBM training loop
"""
WHAT IT FIXES
-------------
pxai/train.py:81

    return F.cross_entropy(logits, y) + concept_loss(c_logit, None)

concept_loss returns c_logit.new_zeros(()) when c_target is None, so the concept term
has been exactly zero for every CBM run in the project. The 16 bottleneck dimensions
are unsupervised latent variables, not the named morphology concepts the docstring
describes. CBM ACCURACY is valid; CBM INTERPRETABILITY is not, and no amount of
attribution fixing changes that -- the concepts have no assigned meaning to attribute.

WHAT IT DOES
------------
  1. loads concepts_v3.csv, one-hot encodes it, aligns rows to loaders.classes
  2. builds the (C, K) target matrix and per-concept pos_weight
  3. passes c_target = table[y] into concept_loss with the weight applied
  4. prints a per-concept BALANCED accuracy report at each validation step

REQUIRED CONFIG CHANGE
----------------------
The encoded width is 23, not 16. Every CBM config needs

    model:
      cbm:
        num_concepts: 23
        concepts_csv: ../Data/Chula-ParasiteEgg-11/concepts_v3.csv

The script rewrites those keys in configs/generated/*cbm*.yaml unless --no-configs.

    python apply_concept_supervision.py --check
    python apply_concept_supervision.py
    python apply_concept_supervision.py --revert
"""

import argparse
import ast
import glob
import os
import shutil
import sys

TRAIN = "pxai/train.py"
CBM = "pxai/models/cbm.py"

EDITS = [
    (TRAIN, "import the loader", True,
     '''from .models.cbm import concept_loss''',
     '''from .models.cbm import concept_loss
from .concepts_loader import (load_concept_table, concept_pos_weight,
                              concept_targets, concept_report,
                              print_concept_report)'''),

    (TRAIN, "build the table", True,
     '''    kind = cfg["model"]["kind"]
    best = 0.0''',
     '''    kind = cfg["model"]["kind"]
    # Class-level concept supervision. Absent csv -> unsupervised bottleneck, i.e.
    # the previous behaviour, but now it says so instead of failing silently.
    ctable = cnames = cweight = None
    if kind == "cbm":
        cpath = cfg["model"]["cbm"].get("concepts_csv")
        if cpath:
            ctable, cnames = load_concept_table(cpath, loaders.classes)
            cweight = concept_pos_weight(ctable).to(device)
            ctable = ctable.to(device)
            k = ctable.shape[1]
            if k != cfg["model"]["cbm"]["num_concepts"]:
                raise ValueError(
                    f"{cpath} encodes {k} concepts but config says "
                    f"{cfg['model']['cbm']['num_concepts']}. Set num_concepts: {k}.")
            print(f"[cbm] {k} class-level concepts from {cpath}; "
                  f"pos_weight {cweight.min():.1f}-{cweight.max():.1f}", flush=True)
        else:
            print("[cbm] no model.cbm.concepts_csv -> bottleneck is UNSUPERVISED. "
                  "Accuracy is valid; interpretability claims are not.", flush=True)
    best = 0.0'''),

    (TRAIN, "pass targets into the loss", True,
     '''                loss = _compute_loss(model, kind, x, y)''',
     '''                loss = _compute_loss(model, kind, x, y, ctable, cweight)'''),

    (TRAIN, "loss signature", True,
     '''def _compute_loss(model, kind, x, y):''',
     '''def _compute_loss(model, kind, x, y, ctable=None, cweight=None):'''),

    (TRAIN, "supervised concept term", True,
     '''    if kind == "cbm":
        feat = model.features(x)
        logits, c_logit = model.head(feat)
        # c_target wiring: plug morphology concept labels here when available
        return F.cross_entropy(logits, y) + concept_loss(c_logit, None)''',
     '''    if kind == "cbm":
        feat = model.features(x)
        logits, c_logit = model.head(feat)
        c_target = concept_targets(ctable, y) if ctable is not None else None
        return F.cross_entropy(logits, y) + concept_loss(c_logit, c_target,
                                                         pos_weight=cweight)'''),

    (TRAIN, "concept report at validation", True,
     '''        acc = evaluate_acc(model, loaders.val, device)
        print(f"epoch {epoch}: val_acc={acc:.4f}")''',
     '''        acc = evaluate_acc(model, loaders.val, device)
        print(f"epoch {epoch}: val_acc={acc:.4f}")
        if ctable is not None and (epoch + 1) % 10 == 0:
            model.eval()
            cl, ct = [], []
            with torch.no_grad():
                for xb, yb in loaders.val:
                    _, cg = model.head(model.features(xb.to(device)))
                    cl.append(cg.float().cpu())
                    ct.append(concept_targets(ctable, yb.to(device)).cpu())
            rows, mb, mr = concept_report(torch.cat(cl), torch.cat(ct), cnames)
            print_concept_report(rows, mb, mr, class_acc=acc)'''),

    (CBM, "pos_weight in concept_loss", True,
     '''def concept_loss(c_logit, c_target, lam: float = 0.5):
    """BCE concept supervision (only when concept labels exist)."""
    if c_target is None:
        return c_logit.new_zeros(())
    return lam * F.binary_cross_entropy_with_logits(c_logit, c_target.float())''',
     '''def concept_loss(c_logit, c_target, lam: float = 0.5, pos_weight=None):
    """BCE concept supervision (only when concept labels exist).

    pos_weight is per concept, n_neg/n_pos. Without it, concepts positive for one
    species out of eleven are learned as "always negative": ~0.91 raw accuracy for a
    model that has learned nothing about that morphology. See concepts_loader.
    """
    if c_target is None:
        return c_logit.new_zeros(())
    return lam * F.binary_cross_entropy_with_logits(
        c_logit, c_target.float(),
        pos_weight=None if pos_weight is None else pos_weight.to(c_logit.dtype))'''),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--csv", default="../Data/Chula-ParasiteEgg-11/concepts_v3.csv")
    ap.add_argument("--num-concepts", type=int, default=23)
    ap.add_argument("--no-configs", action="store_true")
    a = ap.parse_args()

    files = sorted({e[0] for e in EDITS})

    if a.revert:
        for f in files:
            if os.path.exists(f + ".bak-concepts"):
                shutil.copy2(f + ".bak-concepts", f)
                print(f"restored {f}")
        return

    for f in files:
        if not os.path.exists(f):
            sys.exit(f"not found: {f} (run from the repo root)")
    if "concepts_loader" in open(TRAIN).read():
        sys.exit("already patched. --revert first to redo.")

    out, bad = {}, []
    for path in files:
        out[path] = open(path).read()
    for path, name, once, old, new in EDITS:
        n = out[path].count(old)
        if n != 1:
            bad.append((path, name, n, old))
            print(f"  {'MISS ' if n == 0 else 'AMBIG'} {path}: {name} ({n} matches)")
            continue
        out[path] = out[path].replace(old, new, 1)
        print(f"  ok    {path}: {name}")

    if bad:
        print(f"\n{len(bad)} edit(s) did not apply. Nothing written.")
        for path, name, n, old in bad:
            print(f"\n--- {path}: {name}, expected ---\n{old}")
        sys.exit(1)

    for path, src in out.items():
        try:
            ast.parse(src)
        except SyntaxError as e:
            sys.exit(f"\n{path} would not parse: {e}\nNothing written.")
    print("\n  both files parse OK")

    if a.check:
        print("\n--check: nothing written.")
        return

    for path, src in out.items():
        shutil.copy2(path, path + ".bak-concepts")
        open(path, "w").write(src)
        print(f"  patched {path}  (backup {path}.bak-concepts)")

    if not a.no_configs:
        import re
        touched = 0
        for cfgp in glob.glob("configs/generated/*cbm*.yaml"):
            s = open(cfgp).read()
            if "num_concepts" not in s:
                continue
            s2 = re.sub(r"num_concepts:\s*\d+", f"num_concepts: {a.num_concepts}", s)
            if "concepts_csv" not in s2:
                s2 = s2.replace(f"num_concepts: {a.num_concepts}",
                                f"num_concepts: {a.num_concepts}\n"
                                f"    concepts_csv: {a.csv}")
            if s2 != s:
                open(cfgp, "w").write(s2)
                touched += 1
        print(f"  updated {touched} cbm config(s) -> num_concepts {a.num_concepts}")
        print("  CHECK the indentation of the inserted concepts_csv line; it assumes")
        print("  num_concepts sits at 4 spaces under model.cbm.")

    print("""
NEXT
  1. cp concepts_v3.csv ../Data/Chula-ParasiteEgg-11/
  2. python -c "import pxai.train; print('imports OK')"
  3. head -30 configs/generated/roi477_cbm_120ep.yaml     # verify the yaml
  4. train.  The old checkpoint is an UNSUPERVISED bottleneck and is not
     comparable -- keep it as runs/roi477_cbm_120ep/best.pt.unsupervised and
     train into a NEW run dir (roi477_cbm_sup_120ep) so both survive.

WHAT TO READ
  The per-concept report prints every 10 epochs. The number that matters is the GAP
  between class accuracy and macro BALANCED concept accuracy. A large positive gap
  means the model gets the species right while getting the morphology wrong -- which
  a hard bottleneck should make impossible, and is therefore a faithfulness result.""")


if __name__ == "__main__":
    main()
