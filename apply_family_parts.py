#!/usr/bin/env python
# apply_family_parts.py -- register kind: family_parts
"""
Additive. `concept_parts`, `cbm`, `protopnet*` and every existing config and checkpoint
are untouched, so the three arms below are directly comparable against the measured
`concept_parts` baseline.

WHY: the measured failure of concept_parts
------------------------------------------
Above the resolution threshold (feature spans >= 1 backbone cell) concept slots became
cross-species CONSISTENT -- 2/12 -> 12/12 -- while remaining anatomically WRONG. Visual
confirmation on the DINOv2 arm:

    operculum (a polar lid)   -> attends to the egg INTERIOR
    contents=unembryonated    -> attends to the shell RIM and to debris outside the box

Mechanism: concept k is predicted from slot k, and the loss only requires c_k to be
CORRECT. Any region whose features predict c_k is acceptable. `contents=unembryonated` is
true for 5 of 11 species, and Ascaris's shell rim separates those 5 from the other 6 just
as well as its interior does. Twenty-three independent binaries, each satisfiable from
anywhere.

TWO FIXES, ABLATED SEPARATELY
  fix 1  FAMILY-SHARED ATTENTION. One map per family, softmax over the family's mutually
         exclusive values. Covers 19 of 23 concepts. Architectural and dataset-agnostic:
         it is a claim about categorical concept structure, not about parasites, and
         would apply to CUB attributes unchanged.
  fix 2  POLAR PRIOR (w_polar). The 4 binary singletons -- operculum, polar plugs,
         filaments, knob -- get no help from fix 1, and they are the features of
         interest. CDC DPDx states these lie at an extremity, so penalise attention mass
         near the centre. Radius-based, hence rotation-invariant, and needs no box.

    LINE TO HOLD: a prior that encodes a PUBLISHED DEFINITION is legitimate; a prior
    tuned until the maps look right is fitting the answer. If w_polar ends up being
    adjusted because an operculum map is 20px off, the result should not be reported.

FOUR EDITS
  1. models/__init__.py  import build_family_head
  2. models/__init__.py  dispatch on kind == "family_parts"
  3. train.py            concept table setup must fire for family_parts too, else
                         ctable is None and concept supervision is silently ZERO
  4. train.py            loss branch: ce + concept BCE/CE + part_loss

    python apply_family_parts.py --check | --revert

Prerequisite: cp family_parts.py pxai/models/
"""

import argparse
import ast
import os
import shutil
import sys

FILES = ["pxai/models/__init__.py", "pxai/train.py"]

EDITS = [
    ("pxai/models/__init__.py", "import",
     "from .concept_parts import ConceptPartHead, families_from_csv",
     "from .concept_parts import ConceptPartHead, families_from_csv\n"
     "from .family_parts import build_family_head"),

    ("pxai/models/__init__.py", "dispatch",
     '        elif self.kind == "concept_parts":',
     '        elif self.kind == "family_parts":\n'
     '            p = cfg["model"].get("family_parts",\n'
     '                                 cfg["model"].get("concept_parts", {}))\n'
     '            self.head = build_family_head(ch, nc, p)\n'
     '        elif self.kind == "concept_parts":'),

    # WITHOUT THIS ctable is None for family_parts, concept supervision is zero, and the
    # run trains fine while testing nothing.
    ("pxai/train.py", "concept table setup",
     '    if kind in ("cbm", "concept_parts"):',
     '    if kind in ("cbm", "concept_parts", "family_parts"):'),

    ("pxai/train.py", "loss branch",
     '    if kind == "concept_parts":',
     '    if kind == "family_parts":\n'
     '        # same concept supervision as concept_parts, so the ONLY difference is\n'
     '        # family-shared attention (+ the optional polar prior)\n'
     '        feat = model.features(x)\n'
     '        logits, c_logit = model.head(feat)\n'
     '        c_target = concept_targets(ctable, y) if ctable is not None else None\n'
     '        return (F.cross_entropy(logits, y)\n'
     '                + concept_loss(c_logit, c_target, pos_weight=cweight)\n'
     '                + model.head.part_loss(feat))\n'
     '    if kind == "concept_parts":'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    if a.revert:
        for f in FILES:
            b = f + ".bak-family"
            if os.path.exists(b):
                shutil.copy2(b, f)
                print(f"restored {f}")
        return

    if not os.path.exists("pxai/models/family_parts.py"):
        sys.exit("pxai/models/family_parts.py not found.\n"
                 "  cp family_parts.py pxai/models/")
    for f in FILES:
        if not os.path.exists(f):
            sys.exit(f"not found: {f} (run from the repo root)")
    if "build_family_head" in open("pxai/models/__init__.py").read():
        sys.exit("already patched. --revert first to redo.")

    src = {f: open(f).read() for f in FILES}
    bad = []
    for f, name, old, new in EDITS:
        n = src[f].count(old)
        if n != 1:
            bad.append((f, name, n, old))
            print(f"  {'MISS ' if n == 0 else 'AMBIG'} {f}: {name} ({n})")
            continue
        src[f] = src[f].replace(old, new, 1)
        print(f"  ok    {f}: {name}")

    if bad:
        print(f"\n{len(bad)} edit(s) failed. Nothing written.")
        for f, name, n, old in bad:
            print(f"\n--- {f}: {name}, expected ---\n{old}")
        sys.exit(1)

    for f, s in src.items():
        try:
            ast.parse(s)
        except SyntaxError as e:
            sys.exit(f"\n{f} would not parse: {e}\nNothing written.")
    print("\n  both files parse OK")
    if a.check:
        print("\n--check: nothing written.")
        return

    for f, s in src.items():
        shutil.copy2(f, f + ".bak-family")
        open(f, "w").write(s)
        print(f"  patched {f}  (backup {f}.bak-family)")

    print("""
NEXT
  python -c "import pxai.train, pxai.models; print('imports OK')"
  python make_family_configs.py
  python -u preflight_learns.py --config configs/generated/roi477_fam_120ep.yaml --device cuda

  Look for `[family] 23 concepts in 9 families (4 polar), sizes [1,5,5,1,1,1,2,2,5]`.
  If the family count is wrong the softmax groups are wrong and nothing downstream is
  interpretable.""")


if __name__ == "__main__":
    main()
