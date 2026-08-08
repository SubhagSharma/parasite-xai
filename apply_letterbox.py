#!/usr/bin/env python
# apply_letterbox.py -- opt-in aspect-preserving resize, default OFF
"""
Adds a `letterbox` flag to the transform builder in pxai/data.py. **Default is False**,
so every existing config, checkpoint and result is bit-identical to before. Only a
config carrying `data.letterbox: true` sees any difference.

WHAT IT FIXES
`transforms.Resize((img_size, img_size))` -- a TWO-TUPLE -- forces a square by
STRETCHING. Object shape is therefore distorted by an amount set entirely by the source
aspect ratio, a 4.84x spread across this dataset's 13 native image sizes (report §3.3).
Egg shape is diagnostic; the distortion is a camera fingerprint.

With the flag on, the image is padded to square FIRST, so the subsequent resize is
isotropic and a circle stays a circle.

APPLIED IN BOTH BRANCHES
train and eval, deliberately. A model trained on letterboxed images and evaluated on
stretched ones would be measuring a distribution shift, not the transform.

    python apply_letterbox.py --check | --revert

Prerequisite: make_letterbox_config.py, which writes pxai/letterbox.py.
"""

import argparse
import ast
import os
import shutil
import sys

TARGET = "pxai/data.py"

OLD = '''    a = {**_DEFAULT_AUG, **(augment or {})}
    gray = [transforms.Grayscale(num_output_channels=3)] if a["to_grayscale"] else []

    if not train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            *gray,
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

    ops = [transforms.Resize((img_size, img_size))]'''

NEW = '''    a = {**_DEFAULT_AUG, **(augment or {})}
    gray = [transforms.Grayscale(num_output_channels=3)] if a["to_grayscale"] else []

    # Resize((S,S)) with a TWO-TUPLE stretches to square, distorting object shape by an
    # amount set by the source aspect ratio -- a 4.84x spread across this dataset's 13
    # native sizes (report SEC 3.3). With letterbox=True the image is padded to square
    # first, so the resize is isotropic. Default False: every existing config is
    # unchanged, byte for byte.
    if letterbox:
        from .letterbox import LetterboxSquare
        pre = [LetterboxSquare(fill=tuple(int(255 * m) for m in _MEAN))]
    else:
        pre = []

    if not train:
        return transforms.Compose([
            *pre,
            transforms.Resize((img_size, img_size)),
            *gray,
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])

    ops = [*pre, transforms.Resize((img_size, img_size))]'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    bak = TARGET + ".bak-letterbox"

    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy2(bak, TARGET)
        print(f"restored {TARGET}")
        return

    if not os.path.exists(TARGET):
        sys.exit(f"not found: {TARGET} (run from the repo root)")
    src = open(TARGET).read()
    if "LetterboxSquare" in src:
        sys.exit("already patched. --revert first to redo.")
    if not os.path.exists("pxai/letterbox.py"):
        sys.exit("pxai/letterbox.py not found -- run make_letterbox_config.py first")

    n = src.count(OLD)
    if n != 1:
        print(f"{'MISS' if n == 0 else 'AMBIGUOUS'}: {n} matches. Nothing written.")
        print("\n--- expected to find ---")
        print(OLD)
        sys.exit(1)
    out = src.replace(OLD, NEW, 1)
    print("  ok    transform branch")

    # add the parameter to the builder signature, and thread it from the config
    import re
    m = re.search(r"def (_?build_transforms?|_tf)\(([^)]*)\)", out)
    if not m:
        print("  WARN  could not find the transform-builder signature; add "
              "`letterbox: bool = False` to it by hand")
    else:
        fn, args = m.group(1), m.group(2)
        if "letterbox" not in args:
            out = out.replace(m.group(0), f"def {fn}({args}, letterbox: bool = False)", 1)
            print(f"  ok    added letterbox arg to {fn}()")

    calls = out.count(f"{m.group(1)}(") if m else 0
    print(f"  note  {fn}() is called {calls - 1} time(s); the patch below threads the "
          f"config flag" if m else "")
    if m:
        out = re.sub(
            rf"{fn}\(([^)]*?)train=(True|False)([^)]*)\)",
            lambda mm: f"{fn}({mm.group(1)}train={mm.group(2)}{mm.group(3)}, "
                       f"letterbox=cfg['data'].get('letterbox', False))",
            out)
        print("  ok    threaded cfg['data']['letterbox'] into the call sites")

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
VERIFY THE FLAG IS ACTUALLY THREADED before training -- the call-site rewrite is the
fragile part of this patch:

    grep -n "letterbox" pxai/data.py

You want to see it in the signature, the branch, AND every build call. If a call site
was missed, the flag is silently ignored and you get a 4h duplicate of the existing run.

    python -c "
from pxai.utils import load_config
from pxai.data import build_loaders
c = load_config('configs/generated/whole_blackbox_lb_120ep.yaml'); c['device']='cpu'
l = build_loaders(c)
print([t for t in l.test.dataset.dataset.transform.transforms])"

LetterboxSquare must appear FIRST in that list.""")


if __name__ == "__main__":
    main()
