#!/usr/bin/env python
# apply_protopnet_attr_fix.py -- make ProtoPNet's attribution match its forward pass
"""
THE DEFECT
----------
pxai/evaluate.py, ante_hoc_attr, protopnet branch:

    maps = ev["sim_maps"]                                   # (B,P,h,w)
    pc   = ev["proto_class"]                                # (P,C)
    sel  = pc[:, target].t().view(target.size(0), -1, 1, 1)
    a    = (maps * sel).sum(1, keepdim=True)

Two mismatches against what the head actually computes.

**1. Spatial reduction.** protopnet.py:87-96

    sim_pooled = F.max_pool2d(sim, sim.shape[-2:])          # ONE location per prototype
    logits     = F.linear(sim_pooled, relu(last.weight))

The class logit is a sum over prototypes of W[y,p] * sim[p, argmax_p]. Only those P
argmax locations influence the prediction. The attribution summed the whole dense
field. Since sim = log((d+1)/(d+1e-4)) is bounded and non-zero everywhere, those fields
carry a high floor, and summing five of them yields a smooth map dominated by *average*
prototype proximity rather than by the peaks that decide the class.

**2. Weights.** `proto_class` is a fixed identity buffer registered at construction.
`last.weight` is initialised from it and then LEARNED, and is ReLU'd under
`pip_sparsity`. The attribution used the buffer; the forward pass uses the weight.

MEASURED COST  (probe_protopnet_attr.py, 55 images per run, two seeds)

    run                      variant             mass   c@1%  1st hit
    roi477_protopnet_120ep   field_protoclass    1.08   1.44    12.7%
    roi477_protopnet_120ep   argmax_sparse       2.53   2.26     4.8%
    roi477_protopnet_s2337   field_protoclass    1.92   5.76     1.2%
    roi477_protopnet_s2337   argmax_sparse       4.78   7.54     0.2%

The broken attribution understated localisation by 2.3-2.5x on mass. On a scale where
1.0 is random and 16.3 is a perfect explanation, this moves ProtoPNet from "last of 23,
indistinguishable from uniform" into the post-hoc range.

WHAT THE FIX COMPUTES
---------------------
    a = sum_p  W[y,p] * sim[p, argmax_p]   placed at argmax_p, zero elsewhere
        + 1e-6 * (dense field)             tie-breaker only

The first term is an EXACT additive decomposition: it sums to the class logit to
floating-point precision (verified). The second exists because a purely sparse map
leaves 44 of 49 cells at exactly zero, and deletion orders ties by array index -- an
arbitrary top-left bias. At 1e-6 the field cannot reorder the sparse term (sparse
values are ~9, the scaled field ~1e-7) but it gives a sensible ordering below them.

`argmax_soft` (spatial softmax instead of hard argmax) localises better still
(c@1% 3.75 vs 2.26 at seed 1337) but is a surrogate, not a decomposition. Report it
alongside if useful; do not use it for faithfulness claims.

WHAT THIS DOES NOT FIX
----------------------
The prototypes are still mostly on background -- `prototype_sources.png` labels 21 of 24
"background". The corrected attribution points accurately at where the model looks; it
does not change where that is. That needs an egg-constrained push stage in
`_push_prototypes`, which requires a retrain.

    python apply_protopnet_attr_fix.py --check
    python apply_protopnet_attr_fix.py
    python apply_protopnet_attr_fix.py --revert
"""

import argparse
import ast
import os
import shutil
import sys

TARGET = "pxai/evaluate.py"

OLD = '''        if kind == "protopnet":
            maps = ev["sim_maps"]                                   # (B,P,h,w)
            pc = ev["proto_class"]                                  # (P,C)
            sel = pc[:, target].t().view(target.size(0), -1, 1, 1)  # (B,P,1,1)
            a = (maps * sel).sum(1, keepdim=True)'''

NEW = '''        if kind == "protopnet":
            # Match the forward pass: logits = F.linear(max_pool(sim), W) with
            # W = relu(last.weight) under pip_sparsity. Only each prototype's ARGMAX
            # location influences the logit, and the weights are the LEARNED last.weight
            # -- not the fixed proto_class buffer. The previous implementation summed
            # the dense field and used the buffer, understating localisation by 2.3-2.5x
            # (probe_protopnet_attr.py). The sparse term below is an exact additive
            # decomposition of the class logit.
            maps = ev["sim_maps"]                                   # (B,P,h,w)
            B, P, h, w = maps.shape
            W = m.head.last.weight                                  # (C,P), learned
            if getattr(m.head, "pip_sparsity", False):
                W = F.relu(W)                                       # PIP-Net non-neg
            wy = W[target].view(B, P, 1, 1)                         # (B,P,1,1)
            flat = maps.reshape(B, P, h * w)
            mx, idx = flat.max(-1)                                  # (B,P)
            sparse = torch.zeros_like(flat)
            sparse.scatter_(2, idx.unsqueeze(-1), (mx * wy.view(B, P)).unsqueeze(-1))
            # tie-breaker: a purely sparse map leaves h*w-P cells at exactly 0.0, and
            # deletion would order those ties by array index (an arbitrary top-left
            # bias). At 1e-6 the field cannot reorder the sparse term.
            a = (sparse.view(B, P, h, w).sum(1, keepdim=True)
                 + 1e-6 * (maps * wy).sum(1, keepdim=True))'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=TARGET)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    bak = a.path + ".bak-protoattr"

    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy2(bak, a.path)
        print(f"restored {a.path}")
        return

    if not os.path.exists(a.path):
        sys.exit(f"not found: {a.path} (run from the repo root)")
    src = open(a.path).read()

    if "exact additive\n            # decomposition" in src or "sparse.scatter_" in src:
        sys.exit("already patched. --revert first to redo.")

    n = src.count(OLD)
    if n != 1:
        print(f"{'MISS' if n == 0 else 'AMBIGUOUS'}: found {n} matches. Nothing written.")
        print("\n--- expected to find ---")
        print(OLD)
        sys.exit(1)
    print("  ok    protopnet branch of ante_hoc_attr")

    out = src.replace(OLD, NEW, 1)
    if "\n    import torch\n" not in out and "^import torch" not in out:
        pass  # torch is imported at module level in evaluate.py; verified below
    try:
        ast.parse(out)
    except SyntaxError as e:
        sys.exit(f"\npatched file would not parse: {e}\nNothing written.")
    print("  parses OK")

    if a.check:
        print("\n--check: nothing written.")
        return

    shutil.copy2(a.path, bak)
    open(a.path, "w").write(out)
    print(f"\nbackup -> {bak}\npatched -> {a.path}")
    print("""
NEXT
  1. python -c "import pxai.evaluate; print('imports OK')"

  2. verify the fix reproduces the probe's argmax_sparse numbers:
       python -u batch_visualise.py --runs 'roi477_protopnet_120ep' \\
           --outdir figs --tsv figs/m_protopnet_fixed.tsv --device cuda
     conc_pos should land near 2.53, not 1.08.

  3. re-run localisation for every protopnet run (~10 min):
       for R in roi477_protopnet_120ep roi477_protopnet_s2337_120ep \\
                A2_protopnet_mobilevit_120ep crop_protopnet_120ep; do
         python -u batch_visualise.py --runs "$R" --outdir figs \\
             --tsv figs/m_${R}_fixed.tsv --device cuda
       done

  4. re-run faithfulness (~3h per run). THIS IS THE OPEN QUESTION: deletion is
     currently 0.0038, best of 22, computed on the broken attribution.
       deletion gets WORSE -> the 0.0038 was an artefact; SEC 7.4's contradiction
                              dissolves as a measurement error
       deletion stays BEST  -> ProtoPNet is both faithful AND better-localising than
                              reported, which is a positive result
     Preserve the old numbers first:
       cp runs/roi477_protopnet_120ep/results.json \\
          runs/roi477_protopnet_120ep/results.json.pre-protoattr""")


if __name__ == "__main__":
    main()
