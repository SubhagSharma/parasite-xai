#!/bin/bash
# run_night4.sh — scale-normalised ROI, both windows, + the ConvNeXt x ProtoPNet cell.
#
# Ordered so the DECISIVE result lands by hour ~3. If the box dies at 04:00 you
# still have the answer to "did the unified ROI move the shortcut".
#
# PHASE 0  snapshot + preflight (unstick the stale A1 sentinel)   ~3m
# PHASE 1  extract w477 and w679   (PARALLEL, CPU)                ~1.0h
# PHASE 2  train roi477_blackbox, roi679_blackbox                 ~1.8h
# PHASE 3  accuracy + OCCLUSION on those two   <- THE ANSWER      ~0.2h
# PHASE 4  train protopnet, bcos, cbm on both windows             ~5.5h
# PHASE 5  train crop_cbm                                         ~0.6h
# PHASE 6  train A1_convnext_tiny_120ep (ConvNeXt x ProtoPNet)    ~4.5h
# PHASE 7  accuracy, the remaining seven                          ~0.2h
# PHASE 8  occlusion probe, the remaining seven                   ~0.4h
# PHASE 9  frame-geometry diagnostic on both ROI sets             ~0.3h
#                                        GUARANTEED TOTAL ~14.5h
# PHASE 10 OVERFLOW: whole_cbm + its evals                        +4.2h
# PHASE 11 full faithfulness eval          OFF by default         +3.0h
#
# whole_cbm is last on purpose: 4h of I/O-bound decode for the least informative
# CBM cell (the confounded whole-image set, with a head whose concepts are
# unsupervised anyway). If the night runs long it is the only casualty.
#
# Everything writes to NEW run directories. Guarded by .train_complete sentinels,
# so a kill mid-run resumes cleanly on the next launch.
#
#   chmod +x run_night4.sh          # the execute bit is lost on copy
#   nohup ./run_night4.sh > runs/night4.log 2>&1 &

set -u
cd "$(dirname "$0")"
mkdir -p runs
banner () { echo ""; echo "===== $1 :: $(date) ====="; }

DATA=../Data
LABELS=$DATA/Chula-ParasiteEgg-11/labels.json
SRC=$DATA/Chula-ParasiteEgg-11/data
SF=runs/scale_factors.json
RUN_FAITHFULNESS=0          # set to 1 to add ~3h at the very end

find pxai -path "*__pycache__*" -name "*.pyc" -delete 2>/dev/null

# ------------------------------------------------------------------- helpers
extract_one () {
  OUT=$DATA/chula_roi2_w$1
  if [ -f "$OUT/labels.json" ]; then echo "SKIP extract w$1 - exists"; return 0; fi
  python -u make_unified_roi_v2.py \
      --root "$SRC" --labels "$LABELS" --scale-factors "$SF" \
      --out "$OUT" --window "$1" --size 384 \
      > "runs/extract_w$1.log" 2>&1
  echo "extract w$1 exit $?"
}

train_one () {
  NAME=$1
  if [ ! -f "configs/generated/$NAME.yaml" ]; then
    echo "SKIP train $NAME - NO CONFIG configs/generated/$NAME.yaml"; return 0
  fi
  if [ -f "runs/$NAME/.train_complete" ]; then
    echo "SKIP train $NAME - already completed"; return 0
  fi
  banner "TRAIN $NAME"
  mkdir -p "runs/$NAME"
  python -u -m pxai.train --config "configs/generated/$NAME.yaml" \
      > "runs/$NAME/train.log" 2>&1
  code=$?
  [ $code -eq 0 ] && touch "runs/$NAME/.train_complete" \
                  || echo "TRAIN $NAME FAILED (exit $code)"
  echo "===== $NAME done :: $(date) :: exit $code ====="
}

acc_one () {
  [ -f "runs/$1/best.pt" ] || { echo "SKIP acc $1 - no checkpoint"; return 0; }
  python -u eval_accuracy_only.py --config "configs/generated/$1.yaml" \
      --ckpt "runs/$1/best.pt" 2>&1 | tee -a "runs/$1/accuracy.log"
}

# $1 = run name, $2 = labels.json in THAT dataset's coordinate system
occl_one () {
  [ -f "runs/$1/best.pt" ] || { echo "SKIP occlusion $1 - no checkpoint"; return 0; }
  echo "--- occlusion: $1 ---"
  python -u probe_occlusion.py --config "configs/generated/$1.yaml" \
      --ckpt "runs/$1/best.pt" --labels "$2" --device cpu \
      2>&1 | tail -22 | tee -a "runs/$1/occlusion.log"
}

echo "########## NIGHT 4 STARTED $(date) ##########"

# ------------------------------------------------------------------- phase 0
banner "PHASE 0  snapshot + preflight"
./snapshot_stable.sh "pre_night4_$(date +%Y%m%d_%H%M)" || { echo "SNAPSHOT FAILED"; exit 1; }
[ -f "$SF" ] || { echo "MISSING $SF - run probe_scale_decomposition.py --emit first"; exit 1; }

# runs/A1_convnext_tiny_120ep holds a .train_complete sentinel but no checkpoint --
# a run that died before writing anything. Every guard here and in run_night{1,2,3}.sh
# would skip it forever and say nothing.
if [ -f runs/A1_convnext_tiny_120ep/.train_complete ] \
   && [ ! -f runs/A1_convnext_tiny_120ep/best.pt ]; then
  echo "UNSTICK: removing stale sentinel on A1_convnext_tiny_120ep (no best.pt)"
  rm -f runs/A1_convnext_tiny_120ep/.train_complete
fi

# ------------------------------------------------------------------- phase 1
banner "PHASE 1  extract both ROI windows (parallel)"
extract_one 477 & P1=$!
extract_one 679 & P2=$!
wait $P1 $P2
for W in 477 679; do
  D=$DATA/chula_roi2_w$W
  echo "w$W: $(find "$D" -name '*.jpg' 2>/dev/null | wc -l) images"
  tail -16 "runs/extract_w$W.log" 2>/dev/null
done

banner "PHASE 1b  generate configs"
python -u make_roi_configs.py

# ------------------------------------------------------------------- phase 2-3
banner "PHASE 2  blackbox on both windows"
train_one roi477_blackbox_120ep
train_one roi679_blackbox_120ep

banner "PHASE 3  THE ANSWER — did the unified ROI move the shortcut?"
echo "reference points: whole 0.5700, crops 0.2494, chance 0.0909"
acc_one  roi477_blackbox_120ep
acc_one  roi679_blackbox_120ep
occl_one roi477_blackbox_120ep "$DATA/chula_roi2_w477/labels.json"
occl_one roi679_blackbox_120ep "$DATA/chula_roi2_w679/labels.json"

# ------------------------------------------------------------------- phase 4-6
banner "PHASE 4  remaining heads on both windows"
for W in 477 679; do
  for K in protopnet bcos cbm; do train_one "roi${W}_${K}_120ep"; done
done

banner "PHASE 5  crop CBM"
train_one crop_cbm_120ep

banner "PHASE 6  ConvNeXt x ProtoPNet on whole images (completes the 2x2)"
train_one A1_convnext_tiny_120ep

# ------------------------------------------------------------------- phase 7-9
banner "PHASE 7  accuracy, remaining"
for W in 477 679; do
  for K in protopnet bcos cbm; do acc_one "roi${W}_${K}_120ep"; done
done
acc_one crop_cbm_120ep
acc_one A1_convnext_tiny_120ep

banner "PHASE 8  occlusion, remaining"
for W in 477 679; do
  for K in protopnet bcos cbm; do
    occl_one "roi${W}_${K}_120ep" "$DATA/chula_roi2_w$W/labels.json"
  done
done
occl_one crop_cbm_120ep         "$LABELS"
occl_one A1_convnext_tiny_120ep "$LABELS"

banner "PHASE 9  frame-geometry diagnostic on the ROI sets"
for W in 477 679; do
  echo "--- w$W ---"
  python -u diagnose_frame_geometry.py --root "$DATA/chula_roi2_w$W" 2>&1 | tail -20
done

echo ""
echo "########## GUARANTEED SCOPE COMPLETE $(date) ##########"

# ------------------------------------------------------------------ phase 10-11
banner "PHASE 10  OVERFLOW: whole-image CBM"
train_one whole_cbm_120ep
acc_one   whole_cbm_120ep
occl_one  whole_cbm_120ep "$LABELS"

if [ "$RUN_FAITHFULNESS" = "1" ]; then
  banner "PHASE 11  full faithfulness eval (~3h)"
  N=roi679_protopnet_120ep
  if [ -f "runs/$N/best.pt" ]; then
    python -u -m pxai.evaluate --config "configs/generated/$N.yaml" \
        --ckpt "runs/$N/best.pt" > "runs/$N/eval.log" 2>&1
    echo "eval exit $?"
  fi
fi

banner "ACCURACY TABLE"
cat runs/accuracy_2x2.tsv 2>/dev/null || echo "(none)"
echo ""
echo "########## ALL DONE $(date) ##########"
