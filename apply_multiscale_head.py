#!/usr/bin/env python
# apply_multiscale_head.py -- wire MultiScaleProtoHead in behind kind: protopnet_ms
"""
Additive: `protopnet`, `protopnet_diverse`, and every existing config and result are
untouched.

FIVE EDITS
  1. models/__init__.py   import, and dispatch on kind == "protopnet_ms"
  2. models/__init__.py   forward(): pass the feature PYRAMID, not just the last map
  3. train.py             loss branch (cluster/sep/l1, same weights as protopnet)
  4. train.py             push branch -> push_multiscale (per-stage projection)
  5. train.py             the push epoch check recognises the new kind

WHY EDIT 2 IS THE AWKWARD ONE
`model.features(x)` returns ONE tensor and every head, probe and the existing push
assumes that. The multi-scale head needs the whole pyramid. Rather than change the
`features()` contract -- which would break batch_visualise, probe_gradattr,
probe_prototype_diversity and every figure -- the model's forward() branches on kind and
hands the head a list only for `protopnet_ms`. `features()` keeps returning the last map
for everything else.

CHANNEL DISCOVERY
The head needs each stage's channel count. timm's features_only wrapper exposes
`feature_info.channels()`; the patch reads it at construction and fails loudly if the
requested stage index is out of range, rather than mis-sizing a conv and failing at
step 1 of training.

    python apply_multiscale_head.py --check | --revert

Prerequisite: cp protopnet_multiscale.py pxai/models/
"""

import argparse
import ast
import os
import shutil
import sys

FILES = ["pxai/models/__init__.py", "pxai/train.py"]

EDITS = [
    ("pxai/models/__init__.py", "import",
     "from .protopnet import ProtoHead",
     "from .protopnet import ProtoHead\n"
     "from .protopnet_multiscale import (MultiScaleProtoHead, pyramid_forward,\n"
     "                                   push_multiscale)"),

    ("pxai/models/__init__.py", "dispatch",
     '''        if self.kind == "protopnet_diverse":
            p = cfg["model"].get("protopnet_diverse", cfg["model"]["protopnet"])''',
     '''        if self.kind == "protopnet_ms":
            p = cfg["model"].get("protopnet_ms", {})
            try:
                ch_all = self.backbone.net.feature_info.channels()
            except AttributeError as e:
                raise RuntimeError(
                    "protopnet_ms needs a timm features_only backbone exposing "
                    "feature_info.channels()") from e
            stages = list(p.get("stages", [1, 3]))
            bad = [s for s in stages if s >= len(ch_all)]
            if bad:
                raise ValueError(
                    f"stage index {bad} out of range: the backbone has "
                    f"{len(ch_all)} stages with channels {ch_all}")
            self.head = MultiScaleProtoHead(
                ch_all, nc, stages,
                p.get("protos_per_class_per_stage", [2, 3]),
                p.get("proto_dim", 128), p.get("pip_sparsity", True))
            print(f"[ms] stages {stages} channels "
                  f"{[ch_all[s] for s in stages]} "
                  f"protos/class {p.get('protos_per_class_per_stage', [2, 3])}",
                  flush=True)
        elif self.kind == "protopnet_diverse":
            p = cfg["model"].get("protopnet_diverse", cfg["model"]["protopnet"])'''),

    ("pxai/train.py", "loss branch",
     '''    if kind == "protopnet":
        feat = model.features(x)
        logits = model.head(feat)
        ce = F.cross_entropy(logits, y)
        cluster, sep = model.head.cluster_sep_cost(feat, y)
        l1 = model.head.l1_last()
        return ce + 0.8 * cluster - 0.08 * sep + 1e-4 * l1''',
     '''    if kind == "protopnet":
        feat = model.features(x)
        logits = model.head(feat)
        ce = F.cross_entropy(logits, y)
        cluster, sep = model.head.cluster_sep_cost(feat, y)
        l1 = model.head.l1_last()
        return ce + 0.8 * cluster - 0.08 * sep + 1e-4 * l1
    if kind == "protopnet_ms":
        from .models import pyramid_forward
        feats = pyramid_forward(model.backbone, x)
        ce = F.cross_entropy(model.head(feats), y)
        cluster, sep = model.head.cluster_sep_cost(feats, y)
        # identical weights to the protopnet branch, so any difference is the head
        return ce + 0.8 * cluster - 0.08 * sep + 1e-4 * model.head.l1_last()'''),

]

# The push-epoch condition has two forms, depending on whether
# apply_diverse_head.py has been run. Try both; exactly one must match.
PUSH_EPOCH_VARIANTS = [
    ('''        if kind in ("protopnet", "protopnet_diverse") \\
                and (epoch + 1) % cfg["train"]["proto_push_every"] == 0:''',
     '''        if kind in ("protopnet", "protopnet_diverse", "protopnet_ms") \\
                and (epoch + 1) % cfg["train"]["proto_push_every"] == 0:'''),
    ('''        if kind == "protopnet" and (epoch + 1) % cfg["train"]["proto_push_every"] == 0:''',
     '''        if kind in ("protopnet", "protopnet_ms") \\
                and (epoch + 1) % cfg["train"]["proto_push_every"] == 0:'''),
]

# separate: needs the call site, which differs depending on earlier patches
PUSH_ANCHOR = "            stats = _push_prototypes("
PUSH_GUARD = '''            if kind == "protopnet_ms":
                from .models import push_multiscale
                stats = push_multiscale(model, loaders.train, device, max_batches=pmb)
            else:
                stats = _push_prototypes('''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    if a.revert:
        for f in FILES:
            b = f + ".bak-ms"
            if os.path.exists(b):
                shutil.copy2(b, f)
                print(f"restored {f}")
        return

    if not os.path.exists("pxai/models/protopnet_multiscale.py"):
        sys.exit("pxai/models/protopnet_multiscale.py not found.\n"
                 "  cp protopnet_multiscale.py pxai/models/")
    for f in FILES:
        if not os.path.exists(f):
            sys.exit(f"not found: {f} (run from the repo root)")
    if "MultiScaleProtoHead" in open("pxai/models/__init__.py").read():
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

    for _o, _n in PUSH_EPOCH_VARIANTS:
        if src["pxai/train.py"].count(_o) == 1:
            src["pxai/train.py"] = src["pxai/train.py"].replace(_o, _n, 1)
            print("  ok    pxai/train.py: push epoch check")
            break
    else:
        bad.append(("pxai/train.py", "push epoch check", 0,
                    PUSH_EPOCH_VARIANTS[0][0]))
        print("  MISS  pxai/train.py: push epoch check (neither variant matched)")

    n = src["pxai/train.py"].count(PUSH_ANCHOR)
    if n == 1:
        src["pxai/train.py"] = src["pxai/train.py"].replace(
            PUSH_ANCHOR, PUSH_GUARD, 1)
        print("  ok    pxai/train.py: push branch")
    else:
        bad.append(("pxai/train.py", "push branch", n, PUSH_ANCHOR))
        print(f"  MISS  pxai/train.py: push branch ({n})")

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
        shutil.copy2(f, f + ".bak-ms")
        open(f, "w").write(s)
        print(f"  patched {f}  (backup {f}.bak-ms)")

    print("""
NEXT
  python -c "import pxai.train, pxai.models; print('imports OK')"
  python make_multiscale_configs.py
  python -u preflight_learns.py --config configs/generated/roi477_ms_120ep.yaml --device cuda

  The preflight is the gate. Do NOT queue a training run until it passes -- the
  A1_convnext_tiny lesson was 5.5 h of a model that never left chance.""")


if __name__ == "__main__":
    main()