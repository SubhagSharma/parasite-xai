#!/usr/bin/env python
# apply_ms_forward_fix.py -- the edit apply_multiscale_head.py missed
"""
`apply_multiscale_head.py` patched the constructor, the loss branch and the push, but
NOT `InterpretableModel.forward()`:

    def forward(self, x):
        feat = self.backbone(x)      # ONE tensor
        out = self.head(feat)        # MultiScaleProtoHead expects a LIST of maps
        return out[0] if isinstance(out, tuple) else out

Training would have survived, because the `protopnet_ms` loss branch in train.py calls
`pyramid_forward(model.backbone, x)` itself. But **every evaluation path goes through
forward()** -- eval_accuracy_only, pxai.evaluate, batch_visualise, probe_gradattr,
preflight's validation loop -- and would have indexed into the batch dimension instead
of the pyramid, producing either a shape error or, worse, silent nonsense.

Run this AFTER apply_multiscale_head.py. It is a separate file so you do not have to
revert and re-apply a patch that otherwise landed correctly.

    python apply_ms_forward_fix.py --check | --revert
"""

import argparse
import ast
import os
import shutil
import sys

TARGET = "pxai/models/__init__.py"

OLD = '''    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out[0] if isinstance(out, tuple) else out'''

NEW = '''    def forward(self, x):
        # protopnet_ms attaches prototypes at several backbone depths, so its head
        # takes the whole feature pyramid. Every other head takes the last map, and
        # `features()` is left alone so no probe or figure has to change.
        if self.kind == "protopnet_ms":
            out = self.head(pyramid_forward(self.backbone, x))
        else:
            feat = self.backbone(x)
            out = self.head(feat)
        return out[0] if isinstance(out, tuple) else out'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    bak = TARGET + ".bak-msfwd"

    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy2(bak, TARGET)
        print(f"restored {TARGET}")
        return

    if not os.path.exists(TARGET):
        sys.exit(f"not found: {TARGET} (run from the repo root)")
    src = open(TARGET).read()
    if "pyramid_forward(self.backbone" in src:
        sys.exit("already patched. --revert first to redo.")
    if "MultiScaleProtoHead" not in src:
        sys.exit("run apply_multiscale_head.py first")

    n = src.count(OLD)
    if n != 1:
        print(f"{'MISS' if n == 0 else 'AMBIGUOUS'}: {n} matches. Nothing written.")
        print("\n--- expected to find ---")
        print(OLD)
        sys.exit(1)
    out = src.replace(OLD, NEW, 1)
    print("  ok    forward() passes the pyramid for protopnet_ms")

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
VERIFY -- this is the check that matters, and it costs 20 seconds:

  python -c "
import torch
from pxai.utils import load_config
from pxai.data import build_loaders
from pxai.models import build_model
c = load_config('configs/generated/roi477_ms_2way_120ep.yaml'); c['device']='cpu'
l = build_loaders(c); c['model']['num_classes'] = len(l.classes)
m = build_model(c).eval()
with torch.no_grad():
    o = m(torch.randn(4, 3, 224, 224))
print('forward ->', tuple(o.shape), '  want (4, 11)')"

Anything other than (4, 11) means the head is still being handed the wrong thing.""")


if __name__ == "__main__":
    main()
