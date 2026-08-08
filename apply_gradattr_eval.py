#!/usr/bin/env python
# apply_gradattr_eval.py -- let the faithfulness harness use gradient attribution
"""
Part II showed the gradient read-out recovers localisation (5.4x for ProtoPNet, 5.6x
CBM, 2.2x B-cos). It has never been evaluated on FAITHFULNESS -- deletion, insertion
and sensitivity are still computed against the native feature-grid map.

That is the first thing a reviewer will ask about Part II, and it is currently open.

WHAT THIS DOES
Adds a gradient branch to `ante_hoc_attr` in pxai/evaluate.py, guarded by an
environment variable:

    PXAI_GRADATTR=1   -> a = |d(interpretable intermediate)/dx| . x   at pixel resolution
    unset             -> the existing native map, byte-identical behaviour

An env var rather than a config key, deliberately: every existing config keeps working
unchanged, both attributions stay available in one build, and a single run can be
flipped without touching yaml. Set PXAI_GRADATTR_SMOOTH=1 as well for the SmoothGrad
variant (8 copies, sigma 0.10), which scored best in every Part II table.

THE INTERMEDIATE PER HEAD
    protopnet   sim_pooled[p]         weighted by relu(last.weight)[y,p]
    cbm         c_logit[k]            weighted by W_l[y,k]*c_k*(1-c_k)   (exact chain rule)
    bcos        the class logit       (no components)
    blackbox    not applicable -- ante_hoc_attr is never called

WHY IT NEEDS enable_grad
Quantus calls the attribution inside torch.no_grad() in places. The branch wraps its
own torch.enable_grad(), so the graph exists even when the caller has disabled it.

WHAT TO EXPECT
Unknown, and that is the point. Two coherent outcomes:

  deletion IMPROVES  -> the gradient read-out is better on both axes; Part II gets a
                        second, independent result and the recommendation is unambiguous
  deletion WORSENS   -> localisation and faithfulness genuinely trade off here, which
                        is a more interesting finding than either alone and needs
                        saying plainly

Either way it closes the open question. Preserve the current numbers first:

    for R in roi477_protopnet_120ep roi477_cbm_sup_120ep roi477_bcos_120ep; do
      cp runs/$R/results.json runs/$R/results.json.native-attr
    done

    python apply_gradattr_eval.py --check | --revert
"""

import argparse
import ast
import os
import shutil
import sys

TARGET = "pxai/evaluate.py"

ANCHOR = '''    def fn(m, x, target):
        ev = m.explain(x)'''

REPLACEMENT = '''    def fn(m, x, target):
        # ---- gradient read-out (Part II), opt-in via PXAI_GRADATTR ----------------
        # |d(interpretable intermediate)/dx| . x at PIXEL resolution, instead of the
        # head's feature-grid map upsampled. Escapes the 7x7 grid; keeps the
        # per-component decomposition because the intermediate is per prototype /
        # per concept, not the class logit.
        if os.environ.get("PXAI_GRADATTR") == "1":
            import torch as _t
            n_noise = 8 if os.environ.get("PXAI_GRADATTR_SMOOTH") == "1" else 1
            sigma = 0.10 if n_noise > 1 else 0.0
            total = _t.zeros_like(x[:, :1])
            for _ in range(n_noise):
                with _t.enable_grad():          # Quantus may call us under no_grad
                    xi = x if sigma <= 0 else x + _t.randn_like(x) * sigma
                    xi = xi.clone().detach().requires_grad_(True)
                    head = getattr(m, "head", None)
                    if kind == "protopnet":
                        sp, _s = head._similarities(m.backbone(xi))     # (B,P)
                        w = head.last.weight
                        if getattr(head, "pip_sparsity", False):
                            w = F.relu(w)
                        obj = (sp * w[target]).sum()
                    elif kind == "cbm":
                        feat = m.backbone(xi)
                        c_log = head.concept(head.pool(feat).flatten(1))
                        c = _t.sigmoid(c_log)
                        # exact chain rule through the sigmoid bottleneck
                        w = head.classifier.weight[target] * c * (1.0 - c)
                        obj = (c_log * w.detach()).sum()
                    else:                        # bcos: the class logit itself
                        out = m(xi)
                        out = out[0] if isinstance(out, tuple) else out
                        obj = out.gather(1, target.view(-1, 1)).sum()
                    g, = _t.autograd.grad(obj, xi)
                total = total + (g * xi).abs().sum(1, keepdim=True).detach()
            return total / n_noise
        # ---- native feature-grid read-out (default, unchanged) --------------------
        ev = m.explain(x)'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    bak = TARGET + ".bak-gradattr"

    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy2(bak, TARGET)
        print(f"restored {TARGET}")
        return

    if not os.path.exists(TARGET):
        sys.exit(f"not found: {TARGET} (run from the repo root)")
    src = open(TARGET).read()
    if "PXAI_GRADATTR" in src:
        sys.exit("already patched. --revert first to redo.")

    n = src.count(ANCHOR)
    if n != 1:
        print(f"{'MISS' if n == 0 else 'AMBIGUOUS'}: {n} matches. Nothing written.")
        print("\n--- expected to find ---")
        print(ANCHOR)
        sys.exit(1)
    out = src.replace(ANCHOR, REPLACEMENT, 1)
    if "\nimport os" not in out and "\nimport os," not in out:
        out = out.replace("import json", "import json\nimport os", 1)
        print("  ok    added `import os`")
    print("  ok    gradattr branch in ante_hoc_attr")

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
USE
  # native attribution -- unchanged, no env var
  python -u -m pxai.evaluate --config ... --ckpt ...

  # gradient attribution
  PXAI_GRADATTR=1 PXAI_GRADATTR_SMOOTH=1 python -u -m pxai.evaluate --config ... --ckpt ...

VERIFY THE FLAG ACTUALLY BITES before spending 3h on an eval:
  PXAI_GRADATTR=1 python -u batch_visualise.py --runs roi477_protopnet_120ep --fast \\
      --outdir /tmp/gchk --tsv /tmp/gchk.tsv --device cuda
  conc_pos should read ~5.5, not ~2.5.""")


if __name__ == "__main__":
    main()
