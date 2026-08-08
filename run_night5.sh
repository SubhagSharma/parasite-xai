#!/bin/bash
# run_night5.sh — RQ4 on chula_roi2_w477, then the first faithfulness evaluations.
#
# WHY w477 AND NOT CROPS
# The displaced-mask control (runs/displaced_control.tsv) put the egg-specific
# fraction of the occlusion drop at 94.9-99.2% on w477 and 31.7-50.5% on crops,
# where a 186 px box cannot be placed clear of the egg in a 224 px frame. Crop
# numbers cannot be validated in either direction. w477 is the dataset the evidence
# supports, so RQ4 and every faithfulness number below run on it.
#
# PHASE 0  snapshot + LEARNABILITY PREFLIGHT + eval smoke test      ~15m
# PHASE 1  train roi477_effnetlite0                                 ~0.5h
# PHASE 2  train roi477_ghostnet                                    ~0.5h
# PHASE 3  train roi477_convnext                                    ~2.5h
# PHASE 4  accuracy + occlusion sweep + displaced control, all 3    ~0.3h
# PHASE 5  faithfulness eval, roi477_protopnet                      ~3.0h
# PHASE 6  faithfulness eval, roi477_blackbox                       ~3.0h
#                                          GUARANTEED TOTAL  ~10h
# PHASE 7  OVERFLOW: faithfulness, roi477_cbm                       +3.0h
#
# Phases 1-4 close RQ4: four backbones (mobilevit_xs from night 4, plus these three)
# on identical validated data. Phases 5-6 are the first faithfulness numbers on a
# dataset whose confounds are understood.
#
# B-cos faithfulness is deliberately excluded: ~8h alone because contrib_map is
# recomputed inside both MaxSensitivity and MPRT. Fix the caching first.
#
#   chmod +x run_night5.sh
#   nohup ./run_night5.sh > runs/night5.log 2>&1 &

set -u
cd "$(dirname "$0")"
mkdir -p runs
banner () { echo ""; echo "===== $1 :: $(date) ====="; }

DATA=../Data
ROI=$DATA/chula_roi2_w477
LAB=$ROI/labels.json
NEW="roi477_effnetlite0_120ep roi477_ghostnet_120ep roi477_convnext_120ep"
RUN_OVERFLOW=1

find pxai -path "*__pycache__*" -name "*.pyc" -delete 2>/dev/null

train_one () {
  N=$1
  [ -f "configs/generated/$N.yaml" ] || { echo "SKIP $N - NO CONFIG"; return 0; }
  [ -f "runs/$N/.train_complete" ] && { echo "SKIP $N - done"; return 0; }
  banner "TRAIN $N"
  mkdir -p "runs/$N"
  python -u -m pxai.train --config "configs/generated/$N.yaml" \
      > "runs/$N/train.log" 2>&1
  c=$?
  [ $c -eq 0 ] && touch "runs/$N/.train_complete" || echo "TRAIN $N FAILED ($c)"
  echo "===== $N done :: $(date) :: exit $c ====="
}

evaluate_one () {
  N=$1
  [ -f "runs/$N/best.pt" ] || { echo "SKIP eval $N - no checkpoint"; return 0; }
  [ -f "runs/$N/.eval_complete" ] && { echo "SKIP eval $N - done"; return 0; }
  banner "FAITHFULNESS $N"
  python -u -m pxai.evaluate --config "configs/generated/$N.yaml" \
      --ckpt "runs/$N/best.pt" > "runs/$N/eval.log" 2>&1
  c=$?
  [ $c -eq 0 ] && touch "runs/$N/.eval_complete" || echo "EVAL $N FAILED ($c)"
  echo "===== eval $N :: $(date) :: exit $c ====="
  tail -30 "runs/$N/eval.log"
}

echo "########## NIGHT 5 STARTED $(date) ##########"

banner "PHASE 0a  snapshot"
./snapshot_stable.sh "pre_night5_$(date +%Y%m%d_%H%M)" || { echo "SNAPSHOT FAILED"; exit 1; }
[ -f "$LAB" ] || { echo "MISSING $LAB"; exit 1; }

banner "PHASE 0b  learnability preflight — the A1_convnext lesson"
QUEUE=""
for N in $NEW; do
  if [ ! -f "configs/generated/$N.yaml" ]; then
    echo "  $N: NO CONFIG, dropped"; continue
  fi
  echo "--- $N ---"
  python -u preflight_learns.py --config "configs/generated/$N.yaml" --device cuda
  if [ $? -eq 0 ]; then QUEUE="$QUEUE $N"; else echo "  ** $N DROPPED from tonight"; fi
done
echo "queued:${QUEUE:- NONE}"

banner "PHASE 0c  faithfulness smoke test (1 batch, ~3m, not 3h)"
SMOKE=configs/generated/_smoke_roi477_protopnet.yaml
sed -e 's/^  faithfulness_batches:.*/  faithfulness_batches: 1/' \
    -e 's/^    sensitivity:.*/    sensitivity: 1/' \
    -e 's/^    sanity_check:.*/    sanity_check: 1/' \
    -e 's|^output_dir:.*|output_dir: ./runs/_smoke|' \
    configs/generated/roi477_protopnet_120ep.yaml > "$SMOKE"
mkdir -p runs/_smoke
python -u -m pxai.evaluate --config "$SMOKE" \
    --ckpt runs/roi477_protopnet_120ep/best.pt > runs/_smoke/eval.log 2>&1
SMOKE_OK=$?
echo "smoke exit $SMOKE_OK"
[ $SMOKE_OK -ne 0 ] && tail -25 runs/_smoke/eval.log

banner "PHASE 1-3  RQ4 backbone sweep on w477"
for N in $QUEUE; do train_one "$N"; done

banner "PHASE 4  accuracy, occlusion sweep, displaced control"
for N in $QUEUE; do
  [ -f "runs/$N/best.pt" ] || continue
  python -u eval_accuracy_only.py --config "configs/generated/$N.yaml" \
      --ckpt "runs/$N/best.pt" 2>&1 | tee -a "runs/$N/accuracy.log"
  python -u probe_occlusion_v2.py --config "configs/generated/$N.yaml" \
      --ckpt "runs/$N/best.pt" --labels "$LAB" --device cuda \
      --emit-tsv runs/occlusion_sweep.tsv 2>&1 | tail -18
  python -u probe_displaced_control.py --config "configs/generated/$N.yaml" \
      --ckpt "runs/$N/best.pt" --labels "$LAB" --device cuda \
      --emit-tsv runs/displaced_control.tsv 2>&1 | tail -16
done

if [ $SMOKE_OK -eq 0 ]; then
  banner "PHASE 5  faithfulness — roi477_protopnet"
  evaluate_one roi477_protopnet_120ep
  banner "PHASE 6  faithfulness — roi477_blackbox"
  evaluate_one roi477_blackbox_120ep
else
  banner "PHASE 5-6 SKIPPED — smoke test failed, see runs/_smoke/eval.log"
  echo "Not spending 6h on a harness that cannot complete one batch."
fi

echo ""
echo "########## GUARANTEED SCOPE COMPLETE $(date) ##########"

if [ "$RUN_OVERFLOW" = "1" ] && [ $SMOKE_OK -eq 0 ]; then
  banner "PHASE 7  OVERFLOW: faithfulness — roi477_cbm"
  evaluate_one roi477_cbm_120ep
fi

banner "SUMMARY"
echo "--- RQ4: accuracy by backbone on w477 ---"
cat runs/accuracy_2x2.tsv 2>/dev/null | grep -E "roi477|run" || echo "(none)"
echo ""
echo "--- displaced control (egg-specific fraction) ---"
column -t runs/displaced_control.tsv 2>/dev/null || cat runs/displaced_control.tsv
echo ""
echo "########## ALL DONE $(date) ##########"
