#!/usr/bin/env python
# apply_signed_fix.py -- make attribution SIGN visible, in the figures and the TSV
"""
THE PROBLEM
-----------
batch_visualise.py rendered every method through norm01 -> jet, which maps the most
NEGATIVE value to blue and the most positive to red. And metrics() used np.abs().
Between them, three different situations became indistinguishable:

    egg has no attribution          -> blue
    egg has strong NEGATIVE evidence -> blue
    egg is negative, surround positive, |a| flat -> conc = 1.0, read as "no localisation"

That last one matters. `conc ~ 1.0` is currently ambiguous between a genuinely flat map
and a sign-split map with strong structure. I read crop_bcos's c0.9-1.0 as "no better
than uniform"; it may be "strongly localised, opposite signs inside and outside", which
is a completely different claim about the method.

The mix of conventions in pxai/explainers/posthoc.py makes it worse -- the same colour
means different things per column:

    gradcam       SIGNED       (Captum LayerGradCam defaults relu_attributions=False)
    hirescam      >= 0         (.clamp(min=0))
    integrated_gradients >= 0  (.abs())
    lime, kernelshap  SIGNED   (regression coefficients)
    ours:*        SIGNED

WHAT THIS CHANGES
-----------------
1. COLORMAP. Diverging (bwr) with symmetric limits centred at zero, for every method.
   Blue = negative, white = zero, red = positive, unambiguously and comparably.

2. FIVE NEW TSV COLUMNS.
     pos_share  fraction of total |attr| that is positive. 1.00 = the method is
                non-negative, so sign is not an issue for that row.
     in_mean    mean SIGNED attribution inside the box
     out_mean   mean SIGNED attribution outside it
     conc_pos   concentration using only max(a,0) -- "evidence FOR the class"
     conc_neg   concentration using only min(a,0) -- "evidence AGAINST"

   conc_pos is the honest localisation number. "Where is the model looking" almost
   always means positive evidence, and conc_pos measures exactly that where the old
   conc conflated it with negative evidence.

READING THE RESULT
    in_mean < 0 < out_mean, consistently  -> the inversion is REAL. The model treats
        the egg as evidence against, which is a finding, not a rendering artefact.
    pos_share ~ 1.0 and conc_pos ~ conc    -> no sign effect; the original reading
        stands and the ante-hoc methods genuinely fail to localise.

    python apply_signed_fix.py --check
    python apply_signed_fix.py
    python apply_signed_fix.py --revert
"""

import argparse
import ast
import os
import shutil
import sys

TARGET = "batch_visualise.py"

EDITS = [
    ("metrics: signed decomposition", True,
     '''def metrics(attr, mask):
    """-> frac, area, conc, peak. See the module docstring for why frac alone lies."""
    if mask is None:
        return (float("nan"),) * 4
    a = np.abs(np.asarray(attr, dtype=np.float64))
    tot = a.sum()
    if not np.isfinite(tot) or tot <= 0:
        return (float("nan"),) * 4
    frac = float(a[mask].sum() / tot)
    area = float(mask.mean())
    conc = frac / area if area > 0 else float("nan")
    pk = np.unravel_index(int(np.argmax(a)), a.shape)
    return frac, area, conc, float(bool(mask[pk]))''',
     '''def metrics(attr, mask):
    """-> frac, area, conc, peak, pos_share, in_mean, out_mean, conc_pos, conc_neg.

    frac/conc use |a| and so cannot tell "no attribution" from "strong negative
    attribution". conc_pos restricts to max(a,0) -- evidence FOR the class -- which is
    what "where is the model looking" normally means. conc_neg is the mirror. If a
    method is non-negative, pos_share is 1.0 and conc_pos == conc.
    """
    if mask is None:
        return (float("nan"),) * 9
    s = np.asarray(attr, dtype=np.float64)          # SIGNED
    a = np.abs(s)
    tot = a.sum()
    if not np.isfinite(tot) or tot <= 0:
        return (float("nan"),) * 9
    area = float(mask.mean())
    frac = float(a[mask].sum() / tot)
    conc = frac / area if area > 0 else float("nan")
    pk = np.unravel_index(int(np.argmax(a)), a.shape)
    peak = float(bool(mask[pk]))

    pos, neg = np.clip(s, 0, None), np.clip(-s, 0, None)
    pos_share = float(pos.sum() / tot)
    in_mean = float(s[mask].mean())
    out_mean = float(s[~mask].mean())
    cp = float(pos[mask].sum() / pos.sum() / area) if pos.sum() > 0 and area > 0 \\
        else float("nan")
    cn = float(neg[mask].sum() / neg.sum() / area) if neg.sum() > 0 and area > 0 \\
        else float("nan")
    return frac, area, conc, peak, pos_share, in_mean, out_mean, cp, cn'''),

    ("tsv header", True,
     '''        tsv.write("run\\thead\\tdataset\\tclass\\timage\\tmethod\\tcorrect\\t"
                  "frac\\tarea\\tconc\\tpeak\\n")''',
     '''        tsv.write("run\\thead\\tdataset\\tclass\\timage\\tmethod\\tcorrect\\t"
                  "frac\\tarea\\tconc\\tpeak\\t"
                  "pos_share\\tin_mean\\tout_mean\\tconc_pos\\tconc_neg\\n")'''),

    ("unpack + write the new columns", True,
     '''                        fr, ar, co, pk = metrics(m, box)
                        tsv.write(f"{run}\\t{kind}\\t{os.path.basename(root)}\\t{cname}\\t"
                                  f"{fn}\\t{name}\\t{int(pred == label)}\\t{fr:.4f}\\t"
                                  f"{ar:.4f}\\t{co:.4f}\\t{pk:.0f}\\n")''',
     '''                        fr, ar, co, pk, ps, im_, om, cp, cn = metrics(m, box)
                        tsv.write(f"{run}\\t{kind}\\t{os.path.basename(root)}\\t{cname}\\t"
                                  f"{fn}\\t{name}\\t{int(pred == label)}\\t{fr:.4f}\\t"
                                  f"{ar:.4f}\\t{co:.4f}\\t{pk:.0f}\\t{ps:.4f}\\t"
                                  f"{im_:.6g}\\t{om:.6g}\\t{cp:.4f}\\t{cn:.4f}\\n")'''),

    ("failure row column count", True,
     '''                        tsv.write(f"{run}\\t{kind}\\t{os.path.basename(root)}\\t{cname}\\t"
                                  f"{fn}\\t{name}\\t{int(pred == label)}\\tnan\\tnan\\tnan\\tnan\\n")''',
     '''                        tsv.write(f"{run}\\t{kind}\\t{os.path.basename(root)}\\t{cname}\\t"
                                  f"{fn}\\t{name}\\t{int(pred == label)}\\t"
                                  + "\\t".join(["nan"] * 9) + "\\n")'''),

    ("diverging colormap helper", True,
     '''def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)''',
     '''def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def signed01(a):
    """Map a signed map to [0,1] with 0.0 -> 0.5, using symmetric limits.

    With cmap='bwr': blue = negative, white = zero, red = positive. Comparable across
    methods regardless of whether a given one happens to be non-negative.
    """
    a = np.asarray(a, dtype=np.float64)
    lim = np.percentile(np.abs(a), 99)
    if not np.isfinite(lim) or lim <= 0:
        lim = np.abs(a).max() or 1.0
    return np.clip(a / (2.0 * lim) + 0.5, 0, 1)'''),

    ("render signed, per-class figures", True,
     '''                            ax.imshow(img)
                            ax.imshow(norm01(m), cmap="jet", alpha=0.45)
                            ax.set_title(f"f{fr:.2f} c{co:.1f} "
                                         f"{'*' if pk else ''}", fontsize=6)''',
     '''                            ax.imshow(img)
                            ax.imshow(signed01(m), cmap="bwr", alpha=0.45,
                                      vmin=0, vmax=1)
                            ax.set_title(f"c+{cp:.1f} c-{cn:.1f} p{ps:.2f}"
                                         f"{'*' if pk else ''}", fontsize=5.5)'''),

]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=TARGET)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    bak = a.path + ".bak-signed"

    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy2(bak, a.path)
        print(f"restored {a.path}")
        return

    if not os.path.exists(a.path):
        sys.exit(f"not found: {a.path} (run from the repo root)")
    src = open(a.path).read()
    if "def signed01" in src:
        sys.exit("already patched. --revert first to redo.")

    out, bad = src, []
    for name, once, old, new in EDITS:
        n = out.count(old)
        if n != 1:
            bad.append((name, n, old))
            print(f"  {'MISS ' if n == 0 else 'AMBIG'} {name} ({n} matches)")
            continue
        out = out.replace(old, new, 1)
        print(f"  ok    {name}")

    if bad:
        print(f"\n{len(bad)} edit(s) did not apply. Nothing written.")
        for name, n, old in bad:
            print(f"\n--- {name}, expected ---\n{old}")
        sys.exit(1)

    try:
        ast.parse(out)
    except SyntaxError as e:
        sys.exit(f"\npatched file would not parse: {e}\nNothing written.")
    print("\n  parses OK")

    if a.check:
        print("\n--check: nothing written.")
        return

    shutil.copy2(a.path, bak)
    open(a.path, "w").write(out)
    print(f"\nbackup -> {bak}\npatched -> {a.path}")
    print("""
NEXT
  The TSV gains 5 columns, so START A NEW FILE -- appending to the old one mixes
  11-column and 16-column rows and every reader will choke:

    mv figs/attribution_metrics.tsv figs/attribution_metrics.tsv.11col
    nohup python -u batch_visualise.py --fast > runs/batch_signed.log 2>&1 &

  --fast (gradcam, hirescam, IG) is enough to answer the sign question and runs in
  ~25 min. hirescam and IG are non-negative by construction, so they are the control:
  their pos_share must come back 1.00. If it does not, the bug is in metrics(), not
  in the models.

READING
    in_mean < 0 < out_mean consistently -> the inversion is REAL: the egg is evidence
        AGAINST the predicted class, and the ring around it is the evidence for.
    pos_share ~ 1.0 and conc_pos ~ conc -> no sign effect; ante-hoc methods genuinely
        fail to localise and the earlier reading stands.""")


if __name__ == "__main__":
    main()
