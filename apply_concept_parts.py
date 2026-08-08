#!/usr/bin/env python
# apply_concept_parts.py -- register the spatially-grounded concept head
"""
Adds `kind: concept_parts`. Additive: `cbm`, `protopnet`, `protopnet_diverse` and
`protopnet_ms` are untouched, and every existing config and checkpoint still works.

WHY THIS HEAD
The current CBM predicts concepts from `self.pool(feat).flatten(1)` -- a GLOBAL AVERAGE.
There is no attention, so there is nothing to localise: the concept `operculum` is
predicted from the whole image, and asking "where is the operculum" has no answer inside
the model. That is why the CBM's native explanation measured 0.51 conc_pos, BELOW chance.

ConceptPartHead gives each concept its own spatial attention map and predicts that
concept from a scalar projection of its OWN slot. The only route to the concept's value
is through its attention map, so the map must find the evidence. The map is then a
named, localised morphology explanation with no gradient trick and no upsampled grid.

Class-level concepts are sufficient supervision: `has_polar_plugs` is 1 for Trichuris
and 0 for Ascaris, so a slot that must produce that distinction has to attend to the
plug. Between-class contrast supplies what per-image annotation would.

FOUR EDITS
  1. models/__init__.py   import ConceptPartHead, families_from_csv
  2. models/__init__.py   dispatch on kind == "concept_parts"
  3. train.py             loss: cross-entropy + concept BCE + part priors
  4. train.py             the concept report already used by the CBM branch

    python apply_concept_parts.py --check | --revert

Prerequisite: cp concept_parts.py pxai/models/
"""

import argparse
import ast
import os
import shutil
import sys

FILES = ["pxai/models/__init__.py", "pxai/train.py"]

EDITS = [
    ("pxai/models/__init__.py", "import",
     "from .cbm import CBMHead",
     "from .cbm import CBMHead\n"
     "from .concept_parts import ConceptPartHead, families_from_csv"),

    ("pxai/models/__init__.py", "dispatch",
     '        elif self.kind == "cbm":',
     '        elif self.kind == "concept_parts":\n'
     '            p = cfg["model"].get("concept_parts", cfg["model"].get("cbm", {}))\n'
     '            cp = p.get("concepts_csv")\n'
     '            fam = families_from_csv(cp) if cp and os.path.exists(cp) else None\n'
     '            if fam:\n'
     '                print(f"[parts] {len(fam)} concepts, "\n'
     '                      f"{len(set(fam))} morphological families", flush=True)\n'
     '            self.head = ConceptPartHead(\n'
     '                ch, nc, p.get("num_concepts", 23), p.get("slot_dim", 128), fam,\n'
     '                p.get("w_compact", 0.1), p.get("w_distinct", 0.1),\n'
     '                p.get("w_presence", 0.1), p.get("bottleneck", True))\n'
     '        elif self.kind == "cbm":'),

    # WITHOUT THIS the concept table is never built for concept_parts, ctable stays
    # None, concept supervision is silently ZERO, and the run trains fine while
    # testing nothing.
    ("pxai/train.py", "concept table setup",
     '    if kind == "cbm":\n'
     '        cpath = cfg["model"]["cbm"].get("concepts_csv")',
     '    if kind in ("cbm", "concept_parts"):\n'
     '        cpath = cfg["model"].get(kind, cfg["model"].get("cbm", {}))'
     '.get("concepts_csv")'),

    # concept_loss is a MODULE-LEVEL function in cbm.py, already imported by train.py
    # -- not a method on the head. Anchored on the CBM loss branch, which is unique;
    # the earlier `protopnet_ms` anchor matched twice.
    ("pxai/train.py", "loss branch",
     '    if kind == "cbm":\n'
     '        feat = model.features(x)\n'
     '        logits, c_logit = model.head(feat)\n'
     '        c_target = concept_targets(ctable, y) if ctable is not None else None\n'
     '        return F.cross_entropy(logits, y) + concept_loss(c_logit, c_target,\n'
     '                                                         pos_weight=cweight)',
     '    if kind == "concept_parts":\n'
     '        # identical concept supervision to the CBM branch, so the two are\n'
     '        # directly comparable; the only difference is that each concept is\n'
     '        # predicted from its OWN attention slot, not from a global average\n'
     '        feat = model.features(x)\n'
     '        logits, c_logit = model.head(feat)\n'
     '        c_target = concept_targets(ctable, y) if ctable is not None else None\n'
     '        return (F.cross_entropy(logits, y)\n'
     '                + concept_loss(c_logit, c_target, pos_weight=cweight)\n'
     '                + model.head.part_loss(feat))\n'
     '    if kind == "cbm":\n'
     '        feat = model.features(x)\n'
     '        logits, c_logit = model.head(feat)\n'
     '        c_target = concept_targets(ctable, y) if ctable is not None else None\n'
     '        return F.cross_entropy(logits, y) + concept_loss(c_logit, c_target,\n'
     '                                                         pos_weight=cweight)'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    if a.revert:
        for f in FILES:
            b = f + ".bak-parts"
            if os.path.exists(b):
                shutil.copy2(b, f)
                print(f"restored {f}")
        return

    if not os.path.exists("pxai/models/concept_parts.py"):
        sys.exit("pxai/models/concept_parts.py not found.\n"
                 "  cp concept_parts.py pxai/models/")
    for f in FILES:
        if not os.path.exists(f):
            sys.exit(f"not found: {f} (run from the repo root)")
    if "ConceptPartHead" in open("pxai/models/__init__.py").read():
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

    if "import os" not in src["pxai/models/__init__.py"].split("\n\n")[0]:
        src["pxai/models/__init__.py"] = "import os\n" + src["pxai/models/__init__.py"]
        print("  ok    added `import os` to models/__init__.py")

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
        shutil.copy2(f, f + ".bak-parts")
        open(f, "w").write(s)
        print(f"  patched {f}  (backup {f}.bak-parts)")

    print("""
NEXT
  python -c "import pxai.train, pxai.models; print('imports OK')"
  python make_parts_config.py
  python -u preflight_learns.py --config configs/generated/roi477_parts_120ep.yaml --device cuda

  Look for `[parts] 23 concepts, N morphological families` -- that confirms the CSV
  loaded. If it is absent, ctable is None and concept supervision is ZERO: the run
  would train fine and test nothing.""")


if __name__ == "__main__":
    main()
