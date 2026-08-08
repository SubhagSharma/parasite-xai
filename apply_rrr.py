#!/usr/bin/env python
# apply_rrr.py -- Right-for-the-Right-Reasons regularisation, opt-in
"""
Adds a `train.rrr_lambda` option. **Default 0**, so every existing config trains
bit-identically. Only a config setting it non-zero changes anything.

WHAT IT ADDS
    L = L_task + lambda * mean over pixels OUTSIDE the annotation box of (d logit_y/dx)^2

Head-agnostic: it constrains the gradient of the class logit, which every head has, so
`protopnet`, `cbm` and `bcos` arms stay directly comparable and the existing
checkpoints remain the controls.

WHY: Part II SEC 5.5 measured a localisation/faithfulness trade-off (deletion 2-8x worse
under the gradient read-out) and offered an interpretation -- that the two metrics only
agree when the model depends on the object. This tests it. Suppress the shortcut, and
the trade-off should CLOSE.

PRIOR WORK: Ross, Hughes & Doshi-Velez (2017), "Right for the Right Reasons", IJCAI.
The method is theirs. What is new is using it as an instrument on a *measured,
control-validated* shortcut and asking whether the explanation gap closes -- which
needs both axes measured, and neither Ross nor GAIN reports both.

THREE EDITS
  1. import build_rrr / make_indexed_loader
  2. build the penalty and, only when it is active, swap in a path-yielding loader
  3. add the term to the loss, and print the task/penalty ratio in the first epoch

THE RATIO PRINT MATTERS. Ross et al. note lambda should make the "right answers" and
"right reasons" terms the same order of magnitude. The patch prints both for the first
20 steps of epoch 0; if the penalty is 100x the task loss, lower lambda before spending
a night on it.

    python apply_rrr.py --check | --revert

Prerequisite: cp rrr_penalty.py pxai/
"""

import argparse
import ast
import os
import shutil
import sys

TARGET = "pxai/train.py"

EDITS = [
    ("import", "from .models import build_model",
     "from .models import build_model\nfrom .rrr_penalty import build_rrr, make_indexed_loader"),

    ("build the penalty", '''    best = 0.0''',
     '''    # Right-for-the-Right-Reasons: penalise input-gradient magnitude outside the
    # annotation box. Returns None (and costs nothing) when train.rrr_lambda is 0.
    rrr = build_rrr(cfg)
    train_loader = make_indexed_loader(loaders.train) if rrr is not None \\
        else loaders.train
    best = 0.0'''),

    ("loop + loss", '''        for x, y in tqdm(loaders.train, desc=f"ep{epoch}", leave=False):''',
     '''        for _batch in tqdm(train_loader, desc=f"ep{epoch}", leave=False):
            if rrr is not None:
                x, y, _paths = _batch
            else:
                x, y = _batch
                _paths = None'''),
]

LOSS_OLD = '''                loss = _compute_loss(model, kind, x, y, ctable, cweight)'''
LOSS_NEW = '''                loss = _compute_loss(model, kind, x, y, ctable, cweight)
            if rrr is not None:
                # outside autocast: the penalty is a function of a gradient, and double
                # backward under fp16 is numerically fragile
                _pen = rrr(model, x, y, _paths)
                if epoch == 0 and rrr.n_hit + rrr.n_miss < 20 * x.shape[0]:
                    print(f"  [rrr] task {float(loss):.4f}  penalty {float(_pen):.4f}  "
                          f"ratio {float(_pen) / max(float(loss), 1e-9):.2f}", flush=True)
                loss = loss + _pen'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    bak = TARGET + ".bak-rrr"

    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy2(bak, TARGET)
        print(f"restored {TARGET}")
        return

    if not os.path.exists(TARGET):
        sys.exit(f"not found: {TARGET} (run from the repo root)")
    if not os.path.exists("pxai/rrr_penalty.py"):
        sys.exit("pxai/rrr_penalty.py not found.\n  cp rrr_penalty.py pxai/")
    src = open(TARGET).read()
    if "build_rrr" in src:
        sys.exit("already patched. --revert first to redo.")

    out, bad = src, []
    for name, old, new in EDITS:
        n = out.count(old)
        if n != 1:
            bad.append((name, n, old))
            print(f"  {'MISS ' if n == 0 else 'AMBIG'} {name} ({n})")
            continue
        out = out.replace(old, new, 1)
        print(f"  ok    {name}")

    n = out.count(LOSS_OLD)
    if n >= 1:
        out = out.replace(LOSS_OLD, LOSS_NEW, 1)
        print(f"  ok    loss term ({n} call site(s), patched the first)")
    else:
        bad.append(("loss term", n, LOSS_OLD))
        print(f"  MISS  loss term ({n})")

    if bad:
        print(f"\n{len(bad)} edit(s) failed. Nothing written.")
        for name, n, old in bad:
            print(f"\n--- {name}, expected ---\n{old}")
        sys.exit(1)

    try:
        ast.parse(out)
    except SyntaxError as e:
        sys.exit(f"\nwould not parse: {e}\nNothing written.")
    print("  parses OK")
    if a.check:
        print("\n--check: nothing written.")
        return

    shutil.copy2(TARGET, bak)
    open(TARGET, "w").write(out)
    print(f"\nbackup -> {bak}\npatched -> {TARGET}")
    print("""
NEXT
  python -c "import pxai.train; print('imports OK')"
  python make_rrr_configs.py
  python -u preflight_learns.py --config configs/generated/roi477_rrr_120ep.yaml --device cuda

WATCH THE FIRST 20 STEPS
  [rrr] task 2.3841  penalty 0.0043  ratio 0.00
  A ratio far below ~0.1 means lambda is too small to bite; far above ~10 means it will
  dominate and accuracy will collapse. Ross et al. set lambda so both terms are the same
  order of magnitude. Adjust train.rrr_lambda and re-run the preflight -- it is 3
  minutes, against 1.5 h for the real run.""")


if __name__ == "__main__":
    main()
