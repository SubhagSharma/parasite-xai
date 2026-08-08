"""apply_sea_patch.py — wire the SEA head into the four files that need it.

    python apply_sea_patch.py            # dry run, prints the diff it would make
    python apply_sea_patch.py --write

Idempotent: running twice is a no-op. Every edit asserts its anchor exists, so
a silent no-op is impossible -- if the file has drifted this fails loudly
instead of pretending to have patched it. Each edit is 1-3 lines.
"""
from __future__ import annotations

import argparse
import os
import sys

EDITS = [
    # (path, anchor, replacement, marker-that-means-already-applied)
    ("pxai/models/__init__.py",
     "from .blackbox import BlackBox",
     "from .blackbox import BlackBox\nfrom .sea import SEANet",
     "from .sea import SEANet"),

    ("pxai/models/__init__.py",
     '''def build_model(cfg):
    if cfg["model"]["kind"] == "blackbox":''',
     '''def build_model(cfg):
    if cfg["model"]["kind"] == "sea":
        return SEANet(cfg)
    if cfg["model"]["kind"] == "blackbox":''',
     'if cfg["model"]["kind"] == "sea":'),

    ("pxai/train.py",
     '''def _compute_loss(model, kind, x, y):
    if kind == "protopnet":''',
     '''def _compute_loss(model, kind, x, y):
    if kind == "sea":
        from .models.sea import sea_loss
        return sea_loss(model, x, y, getattr(model, "loss_w", None))[0]
    if kind == "protopnet":''',
     'if kind == "sea":'),

    # SEA exposes the same "contrib_map" key as the B-cos head, so the existing
    # gather branch is correct as written -- it only needs to be reachable.
    ("pxai/evaluate.py",
     '        elif kind == "bcos":',
     '        elif kind in ("bcos", "sea"):',
     'elif kind in ("bcos", "sea"):'),

    ("pxai/eval/cost.py",
     '"protopnet": 1, "cbm": 1, "bcos": 1, "amortized": 1,',
     '"protopnet": 1, "cbm": 1, "bcos": 1, "sea": 1, "amortized": 1,',
     '"sea": 1,'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(os.path.join(a.root, "pxai", "models", "sea.py")):
        sys.exit("pxai/models/sea.py is missing -- copy it in first, then "
                 "check with: head -2 pxai/models/sea.py")

    fail = False
    for path, anchor, repl, marker in EDITS:
        full = os.path.join(a.root, path)
        src = open(full).read()
        if marker in src:
            print(f"  skip   {path}  (already applied)")
            continue
        if anchor not in src:
            print(f"  ERROR  {path}  anchor not found:\n         {anchor[:70]!r}")
            fail = True
            continue
        if src.count(anchor) != 1:
            print(f"  ERROR  {path}  anchor appears {src.count(anchor)}x, "
                  "expected exactly 1")
            fail = True
            continue
        print(f"  patch  {path}")
        if a.write:
            open(full, "w").write(src.replace(anchor, repl, 1))

    if fail:
        sys.exit("\nNOTHING WAS WRITTEN. Fix the anchors above first.")
    print("\nwritten." if a.write else "\ndry run only -- re-run with --write")


if __name__ == "__main__":
    main()
