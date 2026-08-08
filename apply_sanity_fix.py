#!/usr/bin/env python
# apply_sanity_fix.py -- apply the three sanity_check fixes to pxai/eval/faithfulness.py
"""
WHY A PATCHER AND NOT A MODULE
------------------------------
These are six edits scattered inside run() in faithfulness.py (counter init, the
metric call, the flat guard, the AssertionError branch, the scores loop, the returned
dict). There is no single function to import and override, so this script does the
edits, verifies them, and can undo them.

    python apply_sanity_fix.py --check     # show what would change, touch nothing
    python apply_sanity_fix.py             # apply, writing a .bak
    python apply_sanity_fix.py --revert    # restore from the .bak

WHAT IT FIXES  (evidence: probe_sanity_v2.py, 32 samples, 3 seeds)

    model      flat%     raw rho    |rho|    reported
    protopnet  100.0%   -0.0502    0.3832    0.0000
    bcos         0.0%   +0.0397    0.2638   +0.0397
    blackbox     0.0%   +0.0789    0.0806   +0.0789
    cbm            -    (eval reported -0.6668)

1. SIGNED CANCELLATION. MPRT reports a signed Spearman; weight dependence is |rho|.
   rho = -0.67 tracks the weights as strongly as +0.67. Signs cancel within a method
   (B-cos: -0.088, +0.165, +0.042 -> +0.040, true |rho| 2.5x larger) and invert the
   ranking across methods (CBM's -0.6668 currently scores as the BEST sanity result
   under "lower is better"; it is the worst). Fixed by taking abs PER SAMPLE, before
   any averaging -- doing it at METRIC_DIRECTION alone would leave the within-method
   half of the problem.

2. COLLAPSE REPORTED AS A SCORE. Two paths write a hard 0.0: the flat guard and the
   AssertionError degeneracy branch. Both are correct in themselves, but ProtoPNet
   hits the guard on 100% of samples across all three seeds, so its 0.0000 is the
   correction firing, not MPRT measuring. Now reported as a collapse RATE, with the
   score set to NaN when nothing was actually measured -- which also keeps it out of
   normalised_aggregates() via the existing isfinite filter. Also fixes a unit bug:
   the degeneracy branch counted batches while the flat guard counted samples.

3. SINGLE SEED. B-cos's rho varies by 0.253 across three randomisation seeds --
   larger than the spread between models. Averages over SANITY_SEEDS and reports the
   std. Costs ~3x the sanity_check time (~15-25 min per model on night-5 timings);
   use --seeds 1 to skip.

NOT FIXED, AND NOT FIXABLE HERE
-------------------------------
sanity_check is not comparable across heads. ProtoPNet's explanation collapses on a
randomised model by construction, so it cannot fail. CBM's attribution only reaches
its head after cbm_attr_patch. B-cos and blackbox give genuine spatial measurements.
Three regimes in one column -- state the regime per model rather than ranking them.
"""

import argparse
import os
import re
import shutil
import sys

TARGET = "pxai/eval/faithfulness.py"

# (name, must_appear_once, old, new)
EDITS = [
    ("1. seed constant", True,
     '''METRIC_DIRECTION: Dict[str, str] = {''',
     '''# Randomisation seeds averaged for sanity_check. MPRT uses one; measured seed
# spread on B-cos is 0.253, larger than the spread between models, so a point
# estimate is dominated by seed noise. Set to (0,) to restore single-seed behaviour.
SANITY_SEEDS = (0, 1, 2)

METRIC_DIRECTION: Dict[str, str] = {'''),

    ("2. direction note", True,
     '''    "sanity_check": "lower",''',
     '''    # Values are |rho| (abs is taken per sample at the call site), so "lower"
    # means "less weight dependence". Never feed signed correlations through this.
    "sanity_check": "lower",'''),

    ("3. scored counter", True,
     '''    collapses: Dict[str, int] = {m: 0 for m in metric_objs}   # sanity_check degeneracy passes''',
     '''    collapses: Dict[str, int] = {m: 0 for m in metric_objs}   # sanity_check degeneracy passes
    scored: Dict[str, int] = {m: 0 for m in metric_objs}      # samples MPRT actually measured
    sanity_seed_std = None'''),

    ("4. abs + seed averaging", True,
     '''                vals, diag = _postprocess(name, metric(**kw))''',
     '''                if name == "sanity_check":
                    # abs PER SAMPLE before averaging: |rho| is the weight
                    # dependence; signed values cancel into a spurious pass.
                    _runs = []
                    for _s in SANITY_SEEDS:
                        torch.manual_seed(_s)
                        np.random.seed(_s)
                        _r, diag = _postprocess(name, metric(**kw))
                        _runs.append(np.abs(_r))
                    vals = np.mean(_runs, axis=0)
                    if len(_runs) > 1:
                        sanity_seed_std = float(np.mean(np.std(_runs, axis=0)))
                else:
                    vals, diag = _postprocess(name, metric(**kw))'''),

    ("5. count scored samples", True,
     '''                    if flat is not None and flat.any():
                        vals = vals.copy()
                        vals[flat] = 0.0
                        collapses[name] += int(flat.sum())''',
     '''                    n_flat = int(flat.sum()) if flat is not None else 0
                    if flat is not None and flat.any():
                        vals = vals.copy()
                        vals[flat] = 0.0
                        collapses[name] += n_flat
                    scored[name] += int(vals.shape[0]) - n_flat'''),

    ("6. degeneracy counts samples", True,
     '''                    acc[name].extend([0.0] * x_np.shape[0])
                    used[name] += 1
                    collapses[name] += 1''',
     '''                    acc[name].extend([0.0] * x_np.shape[0])
                    used[name] += 1
                    collapses[name] += x_np.shape[0]   # was counting BATCHES'''),

    ("7. NaN when never measured", True,
     '''        scores[m] = float(np.mean(finite)) if finite else float("nan")''',
     '''        scores[m] = float(np.mean(finite)) if finite else float("nan")
        if m == "sanity_check" and collapses.get(m, 0) > 0 and scored.get(m, 0) == 0:
            scores[m] = float("nan")
            print("[faithfulness] sanity_check: explanation collapsed on the "
                  "randomised model for EVERY sample. Reporting NaN, not 0.0 -- "
                  "see sanity_collapse_rate. Degeneracy pass, not a measurement.",
                  flush=True)'''),

    ("8. report collapse rate", True,
     '''        "sanity_collapse_batches": collapses["sanity_check"] if "sanity_check" in collapses else 0,''',
     '''        "sanity_collapse_samples": collapses.get("sanity_check", 0),
        "sanity_scored_samples": scored.get("sanity_check", 0),
        # 1.0 means the reported score is entirely the flat-guard correction and
        # MPRT measured nothing. Report this rate, not the score.
        "sanity_collapse_rate": (
            collapses.get("sanity_check", 0)
            / max(collapses.get("sanity_check", 0) + scored.get("sanity_check", 0), 1)),
        "sanity_seed_std": sanity_seed_std,'''),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=TARGET)
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--revert", action="store_true", help="restore from .bak")
    ap.add_argument("--seeds", type=int, default=3, help="1 skips seed averaging")
    a = ap.parse_args()

    bak = a.path + ".bak"

    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy2(bak, a.path)
        print(f"restored {a.path} from {bak}")
        return

    if not os.path.exists(a.path):
        sys.exit(f"not found: {a.path}  (run from the repo root)")
    src = open(a.path).read()

    if "SANITY_SEEDS" in src:
        sys.exit("already patched (SANITY_SEEDS present). --revert first to redo.")

    print(f"{a.path}: {len(src.splitlines())} lines\n")
    out, applied, missing = src, [], []
    for name, once, old, new in EDITS:
        n = out.count(old)
        if n == 0:
            missing.append((name, old))
            print(f"  MISS  {name}")
            continue
        if once and n > 1:
            missing.append((name, old))
            print(f"  AMBIG {name}: matches {n}x, refusing")
            continue
        out = out.replace(old, new, 1)
        applied.append(name)
        print(f"  ok    {name}")

    if a.seeds == 1:
        out = out.replace("SANITY_SEEDS = (0, 1, 2)", "SANITY_SEEDS = (0,)")
        print("\n  seed averaging disabled (--seeds 1)")

    if missing:
        print(f"\n{len(missing)} edit(s) did not apply. The file differs from the")
        print("version this patcher was written against. Nothing written.")
        for name, old in missing:
            print(f"\n--- {name}, expected to find ---\n{old}")
        sys.exit(1)

    import ast
    try:
        ast.parse(out)
    except SyntaxError as e:
        sys.exit(f"\npatched file does not parse: {e}\nNothing written.")
    print("\n  patched file parses OK")

    if a.check:
        print("\n--check: nothing written.")
        return

    shutil.copy2(a.path, bak)
    with open(a.path, "w") as f:
        f.write(out)
    print(f"\nbackup -> {bak}")
    print(f"patched -> {a.path}  ({len(applied)}/{len(EDITS)} edits)")
    print("""
NEXT
  1. python -c "import pxai.eval.faithfulness; print('imports OK')"
  2. apply the CBM attribution fix too (cbm_attr_patch.py), then re-run all three
     evals ONCE rather than twice -- CBM needs it for the attribution, ProtoPNet and
     B-cos need this patch, ~9h batched.
  3. cp runs/<run>/results.json runs/<run>/results.json.pre-sanityfix2  first.

EXPECTED
  protopnet  0.0000 -> NaN, sanity_collapse_rate 1.00
  bcos      +0.0397 -> ~0.26
  cbm       -0.6668 -> ~0.67   (drops from best to worst)
  blackbox  +0.0789 -> ~0.08   (barely moves; its signs were already consistent)""")


if __name__ == "__main__":
    main()
