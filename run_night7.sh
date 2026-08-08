#!/usr/bin/env bash
# run_night7.sh — attribution fix, egg-constrained push, seed replication, faithfulness.
#
#   ./run_night7.sh              # tonight's scope, in order  (~8.8 h)
#   ./run_night7.sh full         # everything incl. a3 + wholefaith  (~15.5 h)
#   ./run_night7.sh patch        # apply + verify the ProtoPNet attribution fix (~20 m)
#   ./run_night7.sh push         # write + apply the egg-constrained push patch
#   ./run_night7.sh eggtrain     # train the 2 egg-push arms
#   ./run_night7.sh probe        # localisation + top-k on everything new
#   ./run_night7.sh faith        # faithfulness, corrected attribution
#   ./run_night7.sh seeds        # 3 more seeds x 4 roi477 heads
#   ./run_night7.sh a3           # sparsity sweep            -- DEFERRED, see below
#   ./run_night7.sh wholefaith   # whole-image faithfulness  -- DEFERRED, see below
#
# DEFERRED TONIGHT
#   a3 (2.2h)          a second mechanism test. eggtrain is the first and the better
#                      one, and a3 is largely redundant if egg-push moves localisation.
#   wholefaith (4.5h)  fills the cross-epoch gap in report SEC 7.3. Real, but changes
#                      no conclusion: the ante-hoc ordering already holds on both
#                      whole images and the ROI under their respective harnesses.
#   Both are sentinel-guarded, so running them another night recomputes nothing.
#
# Sentinel-guarded throughout: a kill or reboot resumes rather than restarting.
# Refuses to launch if pxai is already running, which is the failure that puts you at
# -P 6 and OOMs.
#
# WHY THIS ORDER
# The egg-push arm (stage 3) is the centrepiece and lands ~2 h in. prototype_sources.png
# shows 21 of 24 prototypes sitting on BACKGROUND: _push_prototypes snaps each prototype
# to the nearest same-class patch anywhere in the image, with nothing preferring the egg.
# The explanations do not point at the egg because the prototypes are not there. Stage 2
# constrains push candidates to patches overlapping the annotation box.
#
#   PREDICTION, recorded before the run: conc_pos should rise above the corrected 2.53.
#   If it does -> a FIX for the localisation failure, not just a diagnosis.
#   If it does not -> the cause is deeper than prototype placement. Also worth knowing.
#
# Two seeds everywhere, because seed variance has overturned three conclusions in this
# project already (head ordering, sanity_check spread 0.253, protopnet conc 1.44 vs 5.76).
#
# BUDGET at the stated -P on a 20 GB A100 MIG slice:
#   patch 0.3h | push 0.2h | eggtrain 1.2h | probe 0.4h | faith 3.0h | seeds 3.7h
#                                                          TONIGHT ~8.8h
#   + a3 2.2h + wholefaith 4.5h                            FULL    ~15.5h
#
# Leaves room to chain run_sea_night2.sh (~7h) inside a 16h window:
#   nohup bash -c './run_night7.sh && ./run_sea_night2.sh' > logs/chain.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")"

PAR_TRAIN=3
PAR_EVAL=2
PAR_PROBE=4
MASTER="logs/night7_$(date +%Y%m%d_%H%M).log"
ROI=../Data/chula_roi2_w477

mkdir -p logs figs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

guard_idle() {
  local n
  n=$(pgrep -fc "pxai.(train|evaluate)" || true)
  if [ "${n:-0}" -gt 0 ]; then
    say "ABORT: $n pxai process(es) already running. Wait, or kill them first."
    pgrep -af "pxai.(train|evaluate)" | tee -a "$MASTER" || true
    exit 1
  fi
}

# ---------------------------------------------------------------- stage: patch
stage_patch() {
  say "=== stage: patch — ProtoPNet attribution ==="

  if [ ! -f apply_protopnet_attr_fix.py ]; then
    say "ABORT: apply_protopnet_attr_fix.py not found in the repo root."
    exit 1
  fi

  if python -c "import re,sys; sys.exit(0 if 'sparse.scatter_' in open('pxai/evaluate.py').read() else 1)"; then
    say "  already patched (sparse.scatter_ present)"
  else
    python apply_protopnet_attr_fix.py --check 2>&1 | tee -a "$MASTER"
    python apply_protopnet_attr_fix.py 2>&1 | tee -a "$MASTER"
  fi
  python -c "import pxai.evaluate; print('  imports OK')" 2>&1 | tee -a "$MASTER"

  say "  verifying the patch reproduces probe_protopnet_attr's argmax_sparse (~2.53)"
  python -u batch_visualise.py --runs 'roi477_protopnet_120ep' --fast \
      --outdir figs --tsv figs/m_protoattr_check.tsv --device cuda \
      >>"$MASTER" 2>&1 || say "  WARN: verification run failed"
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import csv, statistics as st, os
p = "figs/m_protoattr_check.tsv"
if not os.path.exists(p):
    print("  no TSV written"); raise SystemExit
v = [float(r["conc_pos"]) for r in csv.DictReader(open(p), delimiter="\t")
     if r["method"] == "ours:protopnet" and r["conc_pos"] not in ("nan", "")]
if v:
    m = st.mean(v)
    print(f"  conc_pos = {m:.2f}   (was 1.08 broken, expect ~2.53 fixed)  n={len(v)}")
    print("  PATCH CONFIRMED" if m > 2.0 else
          "  ** MISMATCH: patch and probe disagree. Investigate before proceeding.")
PY
  say "=== patch stage finished ==="
}

# ----------------------------------------------------------------- stage: push
stage_push() {
  say "=== stage: push — egg-constrained prototype projection ==="

  if python -c "import sys; sys.exit(0 if 'push_egg_only' in open('pxai/train.py').read() else 1)"; then
    say "  already patched"
  else
    cat > apply_egg_push.py <<'PYEOF'
#!/usr/bin/env python
# apply_egg_push.py -- restrict prototype projection to patches that overlap the egg
"""
THE DEFECT
----------
pxai/train.py `_push_prototypes` snaps each prototype onto the nearest patch from a
same-class image. Class-constrained, but NOT location-constrained: the candidate set is
all h*w spatial positions of every class-c image, including pure background.

    patches = zf_img[img_mask].reshape(-1, D)     # ALL 49 patches per image
    d = ((patches - pv) ** 2).sum(1)
    md, mi = d.min(0)                             # nearest patch, wherever it is

prototype_sources.png shows the outcome: 21 of 24 prototypes are labelled "background".
The model's prototypes are learned pieces of background, so its explanations point at
background -- faithfully. The low conc_pos is a correct report of what the model uses.

Chen et al. do not need this constraint because CUB birds fill the frame. Here the egg
is 2-12% of the image and unconstrained push has nowhere to go but background.

THE FIX
-------
Restrict candidates to patches whose receptive field overlaps the annotation box. A
feature cell counts if ANY of its pixels fall inside the box (adaptive max-pool of the
box mask down to the feature grid). Enabled by `train.push_egg_only: true`; absent or
false reproduces the old behaviour exactly.

Falls back to unconstrained push for any prototype with no in-box candidate, and
reports the count -- a high fallback rate means the constraint is too tight and the
result must be qualified.

    python apply_egg_push.py --check | --revert
"""
import argparse, ast, os, shutil, sys

TARGET = "pxai/train.py"

OLD_SIG = '''def _push_prototypes(model, loader, device, max_batches: int | None = None):'''
NEW_SIG = '''def _push_prototypes(model, loader, device, max_batches: int | None = None,
                     egg_only: bool = False, ann=None, img_size: int = 224,
                     margin: float = 0.20):'''

OLD_CALL = '''            stats = _push_prototypes(model, loaders.train, device, max_batches=pmb)'''
NEW_CALL = '''            _egg = cfg["train"].get("push_egg_only", False)
            stats = _push_prototypes(
                model, loaders.train, device, max_batches=pmb,
                egg_only=_egg, ann=(_push_ann if _egg else None),
                img_size=cfg["data"]["img_size"])'''

OLD_LOOP = '''    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        z = head.add_on(model.features(x))                       # (B,D,H,W)
        B, _, H, W = z.shape
        # per-image patch bank so we can restrict by the image's class label
        zf_img = z.permute(0, 2, 3, 1).reshape(B, H * W, D)      # (B, HW, D)

        # for each class present in this batch, match only ITS prototypes to ITS patches
        for c in torch.unique(y).tolist():
            protos_c = proto_of_class[c]
            if not protos_c:
                continue
            img_mask = (y == c)
            patches = zf_img[img_mask].reshape(-1, D)            # (n_c*HW, D)
            if patches.numel() == 0:
                continue
            for p in protos_c:
                pv = head.prototypes[p].view(1, D)
                d = ((patches - pv) ** 2).sum(1)
                md, mi = d.min(0)
                if md < best_dist[p]:
                    best_dist[p] = md.cpu()
                    best_vec[p] = patches[mi].detach().cpu()'''

NEW_LOOP = '''    # egg_only: iterate the dataset by index so file paths are available for the
    # annotation lookup. The default loader yields (x, y) with no path.
    fell_back = 0
    if egg_only:
        from torch.utils.data import Subset
        from .eval.cropgeom import box_in_crop
        ds = loader.dataset
        _b = ds.dataset if isinstance(ds, Subset) else ds
        while isinstance(_b, Subset):
            _b = _b.dataset
        _idx = list(ds.indices) if isinstance(ds, Subset) else list(range(len(_b.samples)))
        bs = loader.batch_size or 32
        chunks = [_idx[i:i + bs] for i in range(0, len(_idx), bs)]
        if max_batches is not None:
            chunks = chunks[:max_batches]
        stream = (( torch.stack([_b[i][0] for i in ch]),
                    torch.tensor([_b.samples[i][1] for i in ch]),
                    [_b.samples[i][0] for i in ch] ) for ch in chunks)
    else:
        stream = ((x, y, None) for x, y in loader)

    for bi, (x, y, paths) in enumerate(stream):
        if max_batches is not None and bi >= max_batches and not egg_only:
            break
        x = x.to(device)
        y = y.to(device)
        z = head.add_on(model.features(x))                       # (B,D,H,W)
        B, _, H, W = z.shape
        zf_img = z.permute(0, 2, 3, 1).reshape(B, H * W, D)      # (B, HW, D)

        # (B, HW) bool: does this feature cell overlap the annotated egg?
        if egg_only:
            keep = torch.zeros(B, H * W, dtype=torch.bool)
            for bi_, pth in enumerate(paths):
                bm = box_in_crop(pth, ann, img_size, margin, True)
                if bm is None or not bm.any():
                    continue
                t = torch.from_numpy(bm.astype("float32"))[None, None]
                cell = F.adaptive_max_pool2d(t, (H, W))[0, 0] > 0
                keep[bi_] = cell.reshape(-1)
        else:
            keep = torch.ones(B, H * W, dtype=torch.bool)

        for c in torch.unique(y).tolist():
            protos_c = proto_of_class[c]
            if not protos_c:
                continue
            img_mask = (y == c)
            patches = zf_img[img_mask].reshape(-1, D)            # (n_c*HW, D)
            if patches.numel() == 0:
                continue
            kmask = keep[img_mask.cpu()].reshape(-1).to(patches.device)
            cand = patches[kmask] if kmask.any() else patches
            if not kmask.any():
                fell_back += 1
            for p in protos_c:
                pv = head.prototypes[p].view(1, D)
                d = ((cand - pv) ** 2).sum(1)
                md, mi = d.min(0)
                if md < best_dist[p]:
                    best_dist[p] = md.cpu()
                    best_vec[p] = cand[mi].detach().cpu()'''

OLD_RET = '''    return {"pushed": int((~missed).sum()), "missed": int(missed.sum())}'''
NEW_RET = '''    return {"pushed": int((~missed).sum()), "missed": int(missed.sum()),
            "egg_only": bool(egg_only), "fell_back": int(fell_back)}'''

OLD_HDR = '''    kind = cfg["model"]["kind"]'''
NEW_HDR = '''    kind = cfg["model"]["kind"]
    # annotations for egg-constrained push; loaded once, not per push
    _push_ann = None
    if kind == "protopnet" and cfg["train"].get("push_egg_only", False):
        import os as _os
        from .eval.cropgeom import load_coco as _load_coco
        _r = cfg["data"]["root"]
        _lp = _os.path.join(_r, "labels.json")
        if not _os.path.exists(_lp):
            _lp = _os.path.join(_os.path.dirname(_r.rstrip("/")),
                                "Chula-ParasiteEgg-11", "labels.json")
        _push_ann = _load_coco(_lp)
        print(f"[push] egg-constrained projection ON, annotations from {_lp}",
              flush=True)'''

EDITS = [("push signature", OLD_SIG, NEW_SIG),
         ("push call site", OLD_CALL, NEW_CALL),
         ("push candidate loop", OLD_LOOP, NEW_LOOP),
         ("push return dict", OLD_RET, NEW_RET),
         ("annotation load", OLD_HDR, NEW_HDR)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    bak = TARGET + ".bak-eggpush"

    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy2(bak, TARGET); print(f"restored {TARGET}"); return

    src = open(TARGET).read()
    if "push_egg_only" in src:
        sys.exit("already patched. --revert first to redo.")

    out, bad = src, []
    for name, old, new in EDITS:
        n = out.count(old)
        if n != 1:
            bad.append((name, n, old)); print(f"  {'MISS' if n==0 else 'AMBIG'}  {name} ({n})")
            continue
        out = out.replace(old, new, 1); print(f"  ok    {name}")
    if bad:
        print(f"\n{len(bad)} edit(s) failed. Nothing written.")
        for name, n, old in bad:
            print(f"\n--- {name}, expected ---\n{old}")
        sys.exit(1)
    if "import torch.nn.functional as F" not in out:
        out = out.replace("import torch\n", "import torch\nimport torch.nn.functional as F\n", 1)
        print("  ok    added torch.nn.functional import")
    try:
        ast.parse(out)
    except SyntaxError as e:
        sys.exit(f"\nwould not parse: {e}\nNothing written.")
    print("  parses OK")
    if a.check:
        print("\n--check: nothing written."); return
    shutil.copy2(TARGET, bak); open(TARGET, "w").write(out)
    print(f"\nbackup -> {bak}\npatched -> {TARGET}")


if __name__ == "__main__":
    main()
PYEOF
    python apply_egg_push.py --check 2>&1 | tee -a "$MASTER"
    python apply_egg_push.py 2>&1 | tee -a "$MASTER"
  fi
  python -c "import pxai.train; print('  imports OK')" 2>&1 | tee -a "$MASTER"

  say "  writing egg-push configs, seeds 1337 and 2337"
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import yaml, copy
base = yaml.safe_load(open("configs/generated/roi477_protopnet_120ep.yaml"))
for seed, tag in ((1337, ""), (2337, "_s2337")):
    c = copy.deepcopy(base)
    c["seed"] = seed
    c["train"]["push_egg_only"] = True
    name = f"roi477_protopnet_eggpush{tag}_120ep"
    c["output_dir"] = f"./runs/{name}"
    yaml.safe_dump(c, open(f"configs/generated/{name}.yaml", "w"), sort_keys=False)
    print(f"  {name}  seed={seed}  push_egg_only=True")
PY
  say "=== push stage finished ==="
}

# ------------------------------------------------------------ generic trainers
train_list() {
  local par="$1"; shift
  printf '%s\n' "$@" | xargs -P "$par" -I{} bash -c '
    a="{}"
    [ -f "configs/generated/$a.yaml" ] || { echo "  skip  $a (no config)"; exit 0; }
    [ -f "runs/$a/.train_complete" ] && { echo "  skip  $a (trained)"; exit 0; }
    mkdir -p "runs/$a"
    echo "  start $a"
    if python -u -m pxai.train --config "configs/generated/$a.yaml" \
         > "runs/$a/train.log" 2>&1; then
      touch "runs/$a/.train_complete"
      echo "  done  $a  $(grep -o "test_acc=[0-9.]*" runs/$a/train.log | tail -1)"
    else
      echo "  FAIL  $a -- see runs/$a/train.log"
    fi' 2>&1 | tee -a "$MASTER"
}

eval_list() {
  local par="$1"; shift
  printf '%s\n' "$@" | xargs -P "$par" -I{} bash -c '
    a="{}"
    [ -f "runs/$a/best.pt" ] || { echo "  skip  $a (no ckpt)"; exit 0; }
    [ -f "runs/$a/.eval_complete" ] && { echo "  skip  $a (evaluated)"; exit 0; }
    [ -f "runs/$a/results.json" ] && \
      cp "runs/$a/results.json" "runs/$a/results.json.pre-night7"
    echo "  start $a"
    if python -u -m pxai.evaluate --config "configs/generated/$a.yaml" \
         --ckpt "runs/$a/best.pt" > "runs/$a/eval.log" 2>&1; then
      touch "runs/$a/.eval_complete"; echo "  done  $a"
    else
      echo "  FAIL  $a -- see runs/$a/eval.log"
    fi' 2>&1 | tee -a "$MASTER"
}

probe_list() {
  local par="$1"; shift
  printf '%s\n' "$@" | xargs -P "$par" -I{} bash -c '
    a="{}"
    [ -f "runs/$a/best.pt" ] || exit 0
    python -u batch_visualise.py --runs "$a" --outdir figs \
        --tsv "figs/m_${a}_n7.tsv" --device cuda > "logs/probe_$a.log" 2>&1 \
      && echo "  vis   $a" || echo "  FAIL vis $a"' 2>&1 | tee -a "$MASTER"
}

# ------------------------------------------------------------- stage: eggtrain
stage_eggtrain() {
  guard_idle
  say "=== stage: eggtrain (2 arms, -P 2) ==="
  ./snapshot_stable.sh pre-night7 >>"$MASTER" 2>&1 || say "WARN: snapshot failed"
  train_list 2 roi477_protopnet_eggpush_120ep roi477_protopnet_eggpush_s2337_120ep
  say "  push diagnostics (fell_back high => constraint too tight, qualify the result):"
  grep -h "\[push\]" runs/roi477_protopnet_eggpush*/train.log 2>/dev/null | tail -8 \
    | tee -a "$MASTER" || true
  say "=== eggtrain finished ==="
}

# ---------------------------------------------------------------- stage: probe
stage_probe() {
  guard_idle
  say "=== stage: probe (-P $PAR_PROBE) ==="
  probe_list "$PAR_PROBE" \
    roi477_protopnet_eggpush_120ep roi477_protopnet_eggpush_s2337_120ep \
    roi477_protopnet_120ep roi477_protopnet_s2337_120ep

  python -u probe_protopnet_attr.py --device cuda \
    --runs roi477_protopnet_eggpush_120ep,roi477_protopnet_eggpush_s2337_120ep,roi477_protopnet_120ep,roi477_protopnet_s2337_120ep \
    2>&1 | tail -20 | tee -a "$MASTER"

  say "  headline comparison: does egg-constrained push move localisation?"
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import csv, glob, collections, statistics as st
d = collections.defaultdict(list)
for p in glob.glob("figs/m_*_n7.tsv") + ["figs/attribution_metrics.tsv"]:
    try:
        for r in csv.DictReader(open(p), delimiter="\t"):
            if r.get("method") != "ours:protopnet":
                continue
            v = float(r.get("conc_pos", "nan"))
            if v == v:
                d[r["run"]].append(v)
    except Exception:
        pass
print(f"  {'run':<44}{'conc_pos':>10}{'n':>6}")
for k in sorted(d, key=lambda k: -st.mean(d[k])):
    print(f"  {k:<44}{st.mean(d[k]):>10.2f}{len(d[k]):>6}")
print("  reference: 1.0 random, 16.3 perfect, IG 12.4, broken attribution 1.08")
PY
  say "=== probe finished ==="
}

# ---------------------------------------------------------------- stage: faith
stage_faith() {
  guard_idle
  say "=== stage: faith — corrected attribution (-P $PAR_EVAL) ==="
  say "  OPEN QUESTION: deletion is 0.0038 (best of 22) on the BROKEN attribution."
  say "  worse now -> it was an artefact; still best -> ProtoPNet is genuinely faithful."
  for a in roi477_protopnet_120ep roi477_protopnet_s2337_120ep; do
    rm -f "runs/$a/.eval_complete"
  done
  eval_list "$PAR_EVAL" roi477_protopnet_120ep roi477_protopnet_s2337_120ep \
      roi477_protopnet_eggpush_120ep roi477_protopnet_eggpush_s2337_120ep
  say "=== faith finished ==="
}

# ---------------------------------------------------------------- stage: seeds
stage_seeds() {
  guard_idle
  say "=== stage: seeds — 3 more seeds x 4 heads (-P $PAR_TRAIN) ==="
  say "  the head ordering failed to replicate at n=2; this takes it to n=5"
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import yaml, copy, os
heads = {"blackbox": "roi477_blackbox_120ep", "protopnet": "roi477_protopnet_120ep",
         "bcos": "roi477_bcos_120ep", "cbm_sup": "roi477_cbm_sup_120ep"}
made = []
for h, src in heads.items():
    p = f"configs/generated/{src}.yaml"
    if not os.path.exists(p):
        print(f"  skip {h}: {p} missing"); continue
    base = yaml.safe_load(open(p))
    for seed in (3337, 4337, 5337):
        c = copy.deepcopy(base); c["seed"] = seed
        n = f"roi477_{h}_s{seed}_120ep"
        c["output_dir"] = f"./runs/{n}"
        yaml.safe_dump(c, open(f"configs/generated/{n}.yaml", "w"), sort_keys=False)
        made.append(n)
print(f"  {len(made)} configs written")
PY
  ARMS=()
  for h in blackbox protopnet bcos cbm_sup; do
    for s in 3337 4337 5337; do ARMS+=("roi477_${h}_s${s}_120ep"); done
  done
  train_list "$PAR_TRAIN" "${ARMS[@]}"
  probe_list "$PAR_PROBE" "${ARMS[@]}"
  say "=== seeds finished ==="
}

# ------------------------------------------------------------------- stage: a3
stage_a3() {
  guard_idle
  say "=== stage: a3 — sparsity sweep (-P $PAR_TRAIN) ==="
  say "  if the localisation limit is SPATIAL, prototype count should not help"
  A3=(A3_proto1 A3_proto3 A3_proto5 A3_proto10 A3_concepts8 A3_concepts16 A3_concepts32)
  train_list "$PAR_TRAIN" "${A3[@]}"
  probe_list "$PAR_PROBE" "${A3[@]}"
  say "=== a3 finished ==="
}

# ----------------------------------------------------------- stage: wholefaith
stage_wholefaith() {
  guard_idle
  say "=== stage: wholefaith — patched harness on whole images (-P $PAR_EVAL) ==="
  say "  no dataset has yet been evaluated under BOTH harnesses; SEC 7.3 is cross-epoch"
  for a in blackbox_mobilevit_120ep A2_protopnet_mobilevit_120ep A2_bcos_mobilevit_120ep; do
    rm -f "runs/$a/.eval_complete"
  done
  eval_list "$PAR_EVAL" blackbox_mobilevit_120ep A2_protopnet_mobilevit_120ep \
      A2_bcos_mobilevit_120ep
  say "=== wholefaith finished ==="
}

# ------------------------------------------------------------------------ main
case "${1:-all}" in
  patch)      stage_patch ;;
  push)       stage_push ;;
  eggtrain)   stage_eggtrain ;;
  probe)      stage_probe ;;
  faith)      stage_faith ;;
  seeds)      stage_seeds ;;
  a3)         stage_a3 ;;
  wholefaith) stage_wholefaith ;;
  all)        stage_patch; stage_push; stage_eggtrain; stage_probe
              stage_faith; stage_seeds
              say "a3 and wholefaith deferred; run './run_night7.sh full' for both" ;;
  full)       stage_patch; stage_push; stage_eggtrain; stage_probe
              stage_faith; stage_seeds; stage_a3; stage_wholefaith ;;
  *)          echo "usage: $0 [all|full|patch|push|eggtrain|probe|faith|seeds|a3|wholefaith]"
              echo "  all  = tonight's scope (~8.8h), a3 and wholefaith deferred"
              echo "  full = everything (~15.5h)"
              exit 2 ;;
esac

say "done. master log: $MASTER"
