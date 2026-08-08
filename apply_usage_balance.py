#!/usr/bin/env python
# apply_usage_balance.py -- fix the winner-take-all dynamic in the cluster loss
"""
THE DIAGNOSIS
=============
Three measurements now bracket the problem:

    backbone rank        after add_on     prototype rank     prototypes / available
    MobileViT   3.16        1.34              1.03                 77%
    DINOv2     11.40        3.26              1.05                 32%

On MobileViT the prototypes use 77% of the rank available to them -- genuinely
rank-limited. **On DINOv2 there are 3.26 directions on offer and the prototypes use
1.05.** Two thirds of the available diversity goes unused.

So prototype collapse is NOT bounded by the representation. Something in the HEAD
collapses them regardless of what is on offer. That is why swapping to a backbone with
5.6x more within-class rank moved prototype rank from 1.03 to 1.05.

THE MECHANISM
=============
pxai/models/protopnet.py, the clustering term:

    cluster = -(sim_pooled * oh).max(1).values.mean()

`oh` masks to the target class's prototypes; `.max(1)` then keeps exactly ONE. Of the
five prototypes per class, precisely one receives clustering gradient per sample -- the
one already winning. The other four get only weak cross-entropy signal through the
linear layer, drift, and are snapped by push onto whatever patch is nearest.

**Nothing ever pulls a non-winning prototype toward a DIFFERENT region.** It is
winner-take-all, and it is independent of how rich the representation is -- exactly the
signature the DINOv2 experiment measured.

Separation has the same structure (`.max(1)` over wrong-class prototypes).

THE FIX, AND WHY NOT THE OBVIOUS ONE
====================================
The obvious fix -- average the clustering term over all same-class prototypes instead of
taking the max -- is WRONG and would make things worse. It would pull every prototype
toward every patch, which is a stronger collapse pressure, not a weaker one. The max is
in Chen et al. for a reason: it encodes "at least one prototype of this class should
match somewhere".

What is missing is not "all prototypes should match" but "**different prototypes should
win on different samples**". That is a load-balancing problem, and it has a standard
solution in the VQ-VAE and mixture-of-experts literature: penalise non-uniform usage of
the codebook.

    u_p = fraction of samples in the batch whose class-p wins the max
    L_usage = KL(u || uniform)   over each class's prototypes

Zero when every prototype of a class wins equally often; large when one dominates.
Differentiable via a soft (temperature-softmax) surrogate for the argmax, so gradient
reaches the losing prototypes.

WHAT THIS PREDICTS
==================
    prototype effective rank rises from 1.03  -> the mechanism is winner-take-all, and
        every published diversity penalty (orthogonality, Hungarian push, sparsity) is a
        workaround for a load-balancing bug. Simple, specific, and a one-term fix.
    rank stays at 1.03  -> not the cause either. Three mechanisms would then be ruled
        out and the honest answer is that the cause is not yet known.

APPLIED TO THE EXISTING DIVERSE HEAD
====================================
`DiverseProtoHead` is already registered, already reaches `diversity_cost()` from
train.py, and already has a config path. Adding the term there avoids new plumbing, and
`w_usage: 0` reproduces current behaviour exactly.

    python apply_usage_balance.py --check | --revert
"""

import argparse
import ast
import os
import shutil
import sys

TARGET = "pxai/models/protopnet_diverse.py"

SIG_OLD = """                 w_orth: float = 0.0, w_sparse: float = 0.0,
                 focal: bool = False, diverse_push: bool = False):"""
SIG_NEW = """                 w_orth: float = 0.0, w_sparse: float = 0.0,
                 focal: bool = False, diverse_push: bool = False,
                 w_usage: float = 0.0, usage_tau: float = 0.5):"""

INIT_OLD = """        self.focal = focal
        self.diverse_push = diverse_push"""
INIT_NEW = """        self.focal = focal
        self.diverse_push = diverse_push
        self.w_usage = w_usage
        self.usage_tau = usage_tau
        self._last_usage = None      # diagnostic: entropy of the winner distribution"""

COST_OLD = '''    def diversity_cost(self):
        """Total diversity penalty. Add to the loss in train.py; zero when both
        weights are zero, so an unconfigured model is unchanged."""
        return self.w_orth * self.orth_cost() + self.w_sparse * self.sparse_cost()'''

COST_NEW = '''    def usage_cost(self, feat, targets):
        """Load balancing: within each class, prototypes should win equally often.

        The clustering term is a max over the class's prototypes, so exactly one gets
        gradient per sample -- winner-take-all. Measured consequence: on a backbone
        offering 3.26 usable directions the prototypes occupy 1.05 (32%), so the
        collapse is not a representation limit.

        A temperature-softmax over the same-class prototypes gives a differentiable
        surrogate for "which one won". Averaged over the batch this is a usage
        distribution; KL to uniform is zero when every prototype wins equally often and
        grows when one dominates. Gradient reaches the LOSING prototypes, which the max
        never does.
        """
        if self.w_usage <= 0 or self.ppc < 2:
            return self.prototypes.new_zeros(())
        pooled, _ = self._similarities(feat)                  # (B,P)
        B = pooled.shape[0]
        tot, n = pooled.new_zeros(()), 0
        ent = []
        for c in torch.unique(targets).tolist():
            sel = (targets == c)
            if sel.sum() < 2:
                continue
            blk = pooled[sel][:, c * self.ppc:(c + 1) * self.ppc]   # (b, ppc)
            soft = F.softmax(blk / self.usage_tau, dim=1)
            u = soft.mean(0).clamp_min(1e-8)                        # usage per prototype
            tot = tot + (u * (u * self.ppc).log()).sum()            # KL(u || uniform)
            n += 1
            ent.append(float(-(u * u.log()).sum()))
        if ent:
            # diagnostic, printed by train.py: log(ppc) = perfectly balanced,
            # 0 = one prototype takes everything
            self._last_usage = sum(ent) / len(ent)
        return tot / max(n, 1)

    def diversity_cost(self, feat=None, targets=None):
        """Total diversity penalty. Zero when every weight is zero, so an unconfigured
        model is unchanged. `feat`/`targets` are needed only for the usage term."""
        c = self.w_orth * self.orth_cost() + self.w_sparse * self.sparse_cost()
        if self.w_usage > 0 and feat is not None and targets is not None:
            c = c + self.w_usage * self.usage_cost(feat, targets)
        return c'''

TRAIN_OLD = """        div = model.head.diversity_cost()"""
TRAIN_NEW = """        div = model.head.diversity_cost(feat, y)
        if getattr(model.head, "_last_usage", None) is not None:
            import math as _m
            _b = _m.log(model.head.ppc)
            if not hasattr(model.head, "_usage_printed"):
                print(f"  [usage] entropy {model.head._last_usage:.3f} / {_b:.3f} "
                      f"({model.head._last_usage / _b:.0%} balanced; 100% = every "
                      f"prototype wins equally)", flush=True)
                model.head._usage_printed = True"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    files = [TARGET, "pxai/train.py"]
    if a.revert:
        for f in files:
            b = f + ".bak-usage"
            if os.path.exists(b):
                shutil.copy2(b, f)
                print(f"restored {f}")
        return

    for f in files:
        if not os.path.exists(f):
            sys.exit(f"not found: {f} (run from the repo root)")
    src = {f: open(f).read() for f in files}
    if "usage_cost" in src[TARGET]:
        sys.exit("already patched. --revert first to redo.")

    bad = []
    for f, name, old, new in ((TARGET, "signature", SIG_OLD, SIG_NEW),
                              (TARGET, "init", INIT_OLD, INIT_NEW),
                              (TARGET, "usage_cost + diversity_cost", COST_OLD, COST_NEW),
                              ("pxai/train.py", "loss call", TRAIN_OLD, TRAIN_NEW)):
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
        shutil.copy2(f, f + ".bak-usage")
        open(f, "w").write(s)
        print(f"  patched {f}  (backup {f}.bak-usage)")

    print("""
NEXT
  python -c "import pxai.train, pxai.models; print('imports OK')"

  python - <<'PY'
import yaml, copy
c = yaml.safe_load(open("configs/generated/roi477_protopnet_120ep.yaml"))
c["model"]["kind"] = "protopnet_diverse"
c["model"]["protopnet_diverse"] = {
    **c["model"].get("protopnet", {}),
    "w_orth": 0.0, "w_sparse": 0.0, "focal": False, "diverse_push": False,
    "w_usage": 1.0, "usage_tau": 0.5}
c["output_dir"] = "./runs/roi477_usage_120ep"
yaml.safe_dump(c, open("configs/generated/roi477_usage_120ep.yaml","w"), sort_keys=False)
print("wrote roi477_usage_120ep.yaml -- usage balancing ONLY, no other diversity term")
PY

  python -u preflight_learns.py --config configs/generated/roi477_usage_120ep.yaml --device cuda

WATCH THE FIRST EPOCH
  [usage] entropy 0.412 / 1.609 (26% balanced; 100% = every prototype wins equally)
  A low starting figure confirms the winner-take-all diagnosis. It should climb during
  training; if it does not, w_usage is too small to bite.

THEN
  python -u -m pxai.train --config configs/generated/roi477_usage_120ep.yaml
  # and the number that decides it -- baseline prototype effective rank is 1.03 of 5""")


if __name__ == "__main__":
    main()
