"""
batch_visualise.py — attribution figures for every run x every class, plus the numbers.

WHAT IT PRODUCES
  figs/<run>/<class>.jpg   5 samples x every explanation method, box drawn in green
  figs/attribution_metrics.tsv    one row per (run, class, image, method)

The TSV is the important output. 25 runs x 11 classes x 5 images x 6 methods is 8k
attributions -- no one reads 275 figures, but the table can be aggregated, sorted and
tested. The figures are for the cases the numbers flag.

THE METRIC FIX
--------------
visualise_attributions.py reported only `frac` = share of attribution mass inside the
box. That is biased two ways and I read the first ProtoPNet figure through it before
noticing:

  * HIGH-FLOOR MAPS ARE PENALISED. ProtoPNet's sim_maps are similarities with a large
    positive offset, never near zero, so mass is spread over the whole frame and the
    box captures a small share wherever the peak actually sits.
  * BOX SIZE IS IGNORED. A Fasciolopsis box covers ~40% of the frame, a Taenia box
    ~8%. frac 0.57 and frac 0.05 are not comparable numbers.

Four columns now, and `conc` is the one to read:

    frac      share of |attribution| inside the box          (as before)
    area      share of the FRAME the box covers
    conc      frac / area -- 1.0 means no better than uniform, <1.0 means the
              attribution actively avoids the egg
    peak      1 if the argmax pixel is inside the box, 0 otherwise -- floor-free,
              so it is the one statistic ProtoPNet's offset cannot distort

Row 4 of the first figure: frac 0.05 on an ~8% box is conc ~0.6, i.e. BELOW uniform.
"frac is low" and "the attribution avoids the egg" are very different claims and only
conc distinguishes them.

USAGE
    python -u batch_visualise.py --fast              # ~25 min, skips lime/kernelshap
    python -u batch_visualise.py                     # ~3-4 h, all six methods
    python -u batch_visualise.py --runs 'roi477_*'   # subset
    python -u batch_visualise.py --no-figs           # TSV only, tiny, fastest

SIZE WARNING
    Full run with figures is roughly 275 JPEGs at ~400 KB = ~110 MB. That is too much
    for a git repo. Commit figs/attribution_metrics.tsv (a few MB) always, and only a
    curated subset of the images. --no-figs then --runs on the interesting ones is the
    cheaper path.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import Subset

from pxai.utils import load_config, pick_device
from pxai.data import build_loaders
from pxai.models import build_model
from pxai.evaluate import ante_hoc_attr
from pxai.explainers.posthoc import explain_posthoc
from pxai.eval.cropgeom import load_coco, box_in_crop

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
ALL_METHODS = ["gradcam", "hirescam", "integrated_gradients", "lime", "kernelshap"]
FAST_METHODS = ["gradcam", "hirescam", "integrated_gradients"]


def denorm(x):
    return np.clip(x.cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)


def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)


def metrics(attr, mask):
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
    return frac, area, conc, float(bool(mask[pk]))


def labels_for(root):
    """Derive the labels.json that matches this dataset's coordinate system.

    chula_roi2_* carry a remapped labels.json inside the dataset directory; using the
    original Chula file against them reads boxes in the wrong coordinate space and
    returns silent garbage.
    """
    local = os.path.join(root, "labels.json")
    if os.path.exists(local):
        return local
    for up in (os.path.join(os.path.dirname(root.rstrip("/")), "Chula-ParasiteEgg-11",
                            "labels.json"),
               os.path.join(root, "..", "labels.json")):
        if os.path.exists(up):
            return os.path.normpath(up)
    return None


def discover(pattern):
    out = []
    for d in sorted(glob.glob(f"runs/{pattern}")):
        run = os.path.basename(d)
        ck, cf = os.path.join(d, "best.pt"), f"configs/generated/{run}.yaml"
        if os.path.isdir(d) and os.path.exists(ck) and os.path.exists(cf):
            out.append((run, cf, ck))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="*")
    ap.add_argument("--n-per-class", type=int, default=5)
    ap.add_argument("--fast", action="store_true", help="skip lime and kernelshap")
    ap.add_argument("--no-figs", action="store_true", help="TSV only")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--tsv", default="figs/attribution_metrics.tsv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--mask-egg", action="store_true",
                    help="blank the box before explaining: renders the shortcut itself")
    a = ap.parse_args()

    methods = FAST_METHODS if a.fast else ALL_METHODS
    runs = discover(a.runs)
    if not runs:
        raise SystemExit(f"no runs matching runs/{a.runs} with best.pt and a config")
    print(f"{len(runs)} run(s), {len(methods)} post-hoc method(s), "
          f"{a.n_per_class} images/class\n")

    os.makedirs(os.path.dirname(a.tsv) or ".", exist_ok=True)
    new = not os.path.exists(a.tsv)
    tsv = open(a.tsv, "a")
    if new:
        tsv.write("run\thead\tdataset\tclass\timage\tmethod\tcorrect\t"
                  "frac\tarea\tconc\tpeak\n")

    t0 = time.time()
    for ri, (run, cfgp, ckpt) in enumerate(runs, 1):
        try:
            cfg = load_config(cfgp)
            cfg["device"] = a.device
            dev = pick_device(cfg["device"])
            S = cfg["data"]["img_size"]
            kind = cfg["model"]["kind"]
            root = cfg["data"]["root"]
            lab = labels_for(root)
            if lab is None:
                print(f"[{ri}/{len(runs)}] {run}: NO labels.json for {root} — skipped")
                continue
            ann = load_coco(lab)
            loaders = build_loaders(cfg)
            classes = loaders.classes
            cfg["model"]["num_classes"] = len(classes)
            model = build_model(cfg).to(dev)
            model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
            model.eval()
        except Exception as e:
            print(f"[{ri}/{len(runs)}] {run}: LOAD FAILED {type(e).__name__}: {e}")
            continue

        ds = loaders.test.dataset
        base = ds.dataset if isinstance(ds, Subset) else ds
        while isinstance(base, Subset):
            base = base.dataset
        idxs = list(ds.indices) if isinstance(ds, Subset) else range(len(base.samples))

        cols = (["ours:" + kind] if kind != "blackbox" else []) + methods
        rng = np.random.default_rng(a.seed)
        print(f"[{ri}/{len(runs)}] {run}  ({kind}, {os.path.basename(root)}, "
              f"{len(cols)} methods)", flush=True)

        for ci, cname in enumerate(classes):
            pool = [i for i in idxs if base.samples[i][1] == ci]
            if not pool:
                continue
            pick = [pool[j] for j in
                    rng.choice(len(pool), min(a.n_per_class, len(pool)), replace=False)]
            fig = axes = None
            if not a.no_figs:
                fig, axes = plt.subplots(len(pick), 1 + len(cols),
                                         figsize=(2.15 * (1 + len(cols)),
                                                  2.45 * len(pick)), squeeze=False)

            for r, gi in enumerate(pick):
                path, label = base.samples[gi]
                x, _ = base[gi]
                x = x.unsqueeze(0).to(dev)
                t = torch.tensor([label], device=dev)
                box = box_in_crop(path, ann, S, a.margin, True)
                if a.mask_egg and box is not None:
                    for c, mv in enumerate(MEAN):
                        x[0, c][torch.from_numpy(box).to(dev)] = float(mv)
                with torch.no_grad():
                    pred = model(x).argmax(1).item()
                img = denorm(x[0])
                fn = os.path.basename(path)

                if axes is not None:
                    ax = axes[r][0]
                    ax.imshow(img)
                    if box is not None:
                        ys, xs = np.nonzero(box)
                        ax.add_patch(Rectangle((xs.min(), ys.min()), np.ptp(xs),
                                               np.ptp(ys), fill=False, ec="lime", lw=1.5))
                    ax.set_title(("OK" if pred == label else
                                  f"-> {classes[pred][:11]}"), fontsize=6)
                    ax.axis("off")
                    if r == 0:
                        ax.text(0.5, 1.22, "input", transform=ax.transAxes,
                                ha="center", fontsize=7, weight="bold")

                for c, name in enumerate(cols):
                    try:
                        with torch.enable_grad():
                            at = (ante_hoc_attr(kind)(model, x, t)
                                  if name.startswith("ours:")
                                  else explain_posthoc(name, model, x, t)[0])
                        m = at.detach().float().cpu().numpy()
                        m = m[0, 0] if m.ndim == 4 else m.squeeze()
                        fr, ar, co, pk = metrics(m, box)
                        tsv.write(f"{run}\t{kind}\t{os.path.basename(root)}\t{cname}\t"
                                  f"{fn}\t{name}\t{int(pred == label)}\t{fr:.4f}\t"
                                  f"{ar:.4f}\t{co:.4f}\t{pk:.0f}\n")
                        if axes is not None:
                            ax = axes[r][c + 1]
                            ax.imshow(img)
                            ax.imshow(norm01(m), cmap="jet", alpha=0.45)
                            ax.set_title(f"f{fr:.2f} c{co:.1f} "
                                         f"{'*' if pk else ''}", fontsize=6)
                    except Exception as e:
                        tsv.write(f"{run}\t{kind}\t{os.path.basename(root)}\t{cname}\t"
                                  f"{fn}\t{name}\t{int(pred == label)}\tnan\tnan\tnan\tnan\n")
                        if axes is not None:
                            ax = axes[r][c + 1]
                            ax.imshow(img)
                            ax.set_title(f"FAIL {type(e).__name__}", fontsize=5,
                                         color="red")
                    if axes is not None:
                        if box is not None:
                            ys, xs = np.nonzero(box)
                            axes[r][c + 1].add_patch(
                                Rectangle((xs.min(), ys.min()), np.ptp(xs), np.ptp(ys),
                                          fill=False, ec="lime", lw=1.1))
                        axes[r][c + 1].axis("off")
                        if r == 0:
                            axes[r][c + 1].text(
                                0.5, 1.22, name.replace("integrated_gradients", "IG"),
                                transform=axes[r][c + 1].transAxes, ha="center",
                                fontsize=7, weight="bold")

            if fig is not None:
                fig.suptitle(f"{run} — {cname}   f=mass-in-box  c=concentration "
                             f"(1.0=uniform)  *=peak in box"
                             + ("  [EGG MASKED]" if a.mask_egg else ""),
                             fontsize=8, y=0.999)
                fig.tight_layout(rect=[0, 0, 1, 0.985])
                d = os.path.join(a.outdir, run)
                os.makedirs(d, exist_ok=True)
                safe = cname.replace(" ", "_").replace(".", "")
                fig.savefig(os.path.join(d, f"{safe}.jpg"), dpi=a.dpi,
                            bbox_inches="tight", pil_kwargs={"quality": 85})
                plt.close(fig)
            tsv.flush()
        print(f"    done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)

    tsv.close()
    print(f"\nTSV -> {a.tsv}")
    print("""
READ THE TSV FIRST, NOT THE FIGURES

  # mean concentration per method, all runs  (1.0 = no better than uniform)
  python - <<'PY'
import csv, collections, statistics as st
d=collections.defaultdict(list); p=collections.defaultdict(list)
for r in csv.DictReader(open('figs/attribution_metrics.tsv'), delimiter='\\t'):
    try: c=float(r['conc']); k=float(r['peak'])
    except ValueError: continue
    if c==c: d[(r['head'],r['method'])].append(c); p[(r['head'],r['method'])].append(k)
print(f"{'head':<11}{'method':<22}{'conc':>8}{'peak%':>8}{'n':>7}")
for k in sorted(d, key=lambda k:-st.mean(d[k])):
    print(f"{k[0]:<11}{k[1]:<22}{st.mean(d[k]):>8.2f}{st.mean(p[k])*100:>7.0f}%{len(d[k]):>7}")
PY

conc < 1.0 means the attribution puts LESS mass on the egg than a uniform map would.
For an explanation method that is a failure, whatever its deletion score says.""")


if __name__ == "__main__":
    main()
