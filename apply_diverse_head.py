#!/usr/bin/env python
# apply_diverse_head.py -- wire DiverseProtoHead into the factory, loss and push
"""
Registers `kind: protopnet_diverse` without touching ProtoHead, the existing loss for
`kind: protopnet`, or any current result. Everything is additive.

FOUR EDITS
  1. pxai/models/__init__.py   import DiverseProtoHead, dispatch on the new kind
  2. pxai/train.py             loss branch: cluster/sep/l1 as before, PLUS
                               head.diversity_cost()
  3. pxai/train.py             push: recognise the new kind so prototypes are projected
  4. pxai/train.py             push: optional Hungarian assignment for distinct patches

WHY THE LOSS BRANCH IS SEPARATE
The existing branch is `if kind == "protopnet":`. Adding `protopnet_diverse` to that
condition would work, but a separate branch makes the diversity term visible in the
code rather than hidden behind a tuple membership test, and guarantees a `protopnet`
run is bit-identical to before.

DIVERSE PUSH
Standard push projects each prototype independently onto its nearest same-class patch,
so k prototypes can and do collapse onto one patch. With `diverse_push: true` the k
prototypes of a class are assigned to k DISTINCT candidates by linear assignment
(Hungarian), minimising total distance. Candidates are de-duplicated per (image, cell)
first, so two prototypes cannot take the same location of the same image.

    python apply_diverse_head.py --check | --revert

Prerequisite: copy protopnet_diverse.py to pxai/models/ first.
"""

import argparse
import ast
import os
import shutil
import sys

FILES = ["pxai/models/__init__.py", "pxai/train.py"]

EDITS = [
    ("pxai/models/__init__.py", "import",
     '''from .protopnet import ProtoHead''',
     '''from .protopnet import ProtoHead
from .protopnet_diverse import DiverseProtoHead'''),

    ("pxai/models/__init__.py", "dispatch",
     '''        if self.kind == "protopnet":
            p = cfg["model"]["protopnet"]''',
     '''        if self.kind == "protopnet_diverse":
            p = cfg["model"].get("protopnet_diverse", cfg["model"]["protopnet"])
            self.head = DiverseProtoHead(
                ch, nc,
                p.get("num_prototypes_per_class", 5),
                p.get("proto_dim", 128),
                p.get("pip_sparsity", True),
                w_orth=p.get("w_orth", 0.0),
                w_sparse=p.get("w_sparse", 0.0),
                focal=p.get("focal", False),
                diverse_push=p.get("diverse_push", False))
        elif self.kind == "protopnet":
            p = cfg["model"]["protopnet"]'''),

    ("pxai/train.py", "loss branch",
     '''        return ce + 0.8 * cluster - 0.08 * sep + 1e-4 * l1''',
     '''        return ce + 0.8 * cluster - 0.08 * sep + 1e-4 * l1
    if kind == "protopnet_diverse":
        feat = model.features(x)
        logits = model.head(feat)
        ce = F.cross_entropy(logits, y)
        cluster, sep = model.head.cluster_sep_cost(feat, y)
        l1 = model.head.l1_last()
        # identical to the protopnet branch, plus the same-class diversity penalty.
        # diversity_cost() returns exactly 0 when w_orth and w_sparse are both 0, so an
        # unconfigured diverse head trains identically to the original.
        div = model.head.diversity_cost()
        return ce + 0.8 * cluster - 0.08 * sep + 1e-4 * l1 + div'''),

    ("pxai/train.py", "push kind check",
     '''        if kind == "protopnet" and (epoch + 1) % cfg["train"]["proto_push_every"] == 0:''',
     '''        if kind in ("protopnet", "protopnet_diverse") \\
                and (epoch + 1) % cfg["train"]["proto_push_every"] == 0:'''),
]

# Applied separately: needs the surrounding block, which differs if apply_egg_push ran.
PUSH_OLD = '''            for p in protos_c:
                pv = head.prototypes[p].view(1, D)
                d = ((patches - pv) ** 2).sum(1)
                md, mi = d.min(0)
                if md < best_dist[p]:
                    best_dist[p] = md.cpu()
                    best_vec[p] = patches[mi].detach().cpu()'''

PUSH_NEW = '''            if getattr(head, "diverse_push", False) and len(protos_c) > 1:
                # Assign the k prototypes of this class to k DISTINCT patches by
                # linear assignment, instead of each taking its nearest independently
                # (which lets them collapse onto the same patch).
                from .models.protopnet_diverse import assign_diverse
                uniq = torch.unique(patches, dim=0)
                cols = assign_diverse(head.prototypes[protos_c].view(len(protos_c), D),
                                      uniq, len(protos_c))
                for j, p in enumerate(protos_c):
                    if j >= len(cols):
                        break
                    v = uniq[cols[j]]
                    md = ((v - head.prototypes[p].view(D)) ** 2).sum()
                    if md < best_dist[p]:
                        best_dist[p] = md.cpu()
                        best_vec[p] = v.detach().cpu()
            else:
                for p in protos_c:
                    pv = head.prototypes[p].view(1, D)
                    d = ((patches - pv) ** 2).sum(1)
                    md, mi = d.min(0)
                    if md < best_dist[p]:
                        best_dist[p] = md.cpu()
                        best_vec[p] = patches[mi].detach().cpu()'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--no-diverse-push", action="store_true",
                    help="skip edit 4 if the push loop has been modified elsewhere")
    a = ap.parse_args()

    if a.revert:
        for f in FILES:
            b = f + ".bak-diverse"
            if os.path.exists(b):
                shutil.copy2(b, f)
                print(f"restored {f}")
        return

    if not os.path.exists("pxai/models/protopnet_diverse.py"):
        sys.exit("pxai/models/protopnet_diverse.py not found.\n"
                 "  cp protopnet_diverse.py pxai/models/")
    for f in FILES:
        if not os.path.exists(f):
            sys.exit(f"not found: {f} (run from the repo root)")
    if "DiverseProtoHead" in open("pxai/models/__init__.py").read():
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

    if not a.no_diverse_push:
        n = src["pxai/train.py"].count(PUSH_OLD)
        if n == 1:
            src["pxai/train.py"] = src["pxai/train.py"].replace(PUSH_OLD, PUSH_NEW, 1)
            print("  ok    pxai/train.py: diverse push")
        else:
            print(f"  SKIP  pxai/train.py: diverse push ({n} matches -- the push loop "
                  f"differs, probably because apply_egg_push ran).")
            print("        Everything else applies; diverse_push will be inert.")

    if bad:
        print(f"\n{len(bad)} required edit(s) failed. Nothing written.")
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
        shutil.copy2(f, f + ".bak-diverse")
        open(f, "w").write(s)
        print(f"  patched {f}  (backup {f}.bak-diverse)")

    print("""
NEXT
  python -c "import pxai.train, pxai.models; print('imports OK')"
  python make_diverse_configs.py
  ./run_diversity_ablation.sh""")


if __name__ == "__main__":
    main()
