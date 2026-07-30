#!/bin/bash
# run_night.sh — tonight's jobs, STRICTLY SERIAL (parallel runs corrupt results).
#
# Order is deliberate: the shortest decision-critical job runs first so its answer
# exists by morning even if a later job overruns; the long ConvNeXt train follows.
#
#   0. snapshot everything to stable/  (nothing in runs/ is ever lost)
#   1. shuffle control, full test set   ~1h   decides if infidelity_scaled is usable
#   2. A2 ProtoPNet re-eval             ~2h   3 infidelity variants (no retrain)
#   3. ConvNeXt-Tiny black box, 120 ep  ~8h   budget parity for the accuracy claim
#   4. ConvNeXt eval                    ~2h   7 metrics, 5 post-hoc methods
#   5. B-cos head, 120 ep               ~4h   third A2 head (bonus)
#   6. B-cos eval                       ~2h   bonus
#                                      ~19h total; the first 3h answer the
#                                      infidelity question, the first 13h deliver
#                                      budget parity. Later jobs are upside.
#
# Every existing results.json is BACKED UP before being overwritten, so the current
# (raw-infidelity) numbers stay available for comparison.
#
#   chmod +x run_night.sh
#   nohup ./run_night.sh > runs/night.log 2>&1 &
#
# Morning:  cat runs/night.log        (start/finish times + exit codes)

set -u
cd "$(dirname "$0")"
mkdir -p runs

DEV=cuda            # set to cpu if the GPU is busy (much slower)

banner () { echo ""; echo "===== $1 :: $(date) ====="; }



train_one () {
  NAME=$1
  # Guard on a COMPLETION SENTINEL, not on best.pt. A run killed part-way leaves a
  # best.pt behind, which would otherwise look finished and get silently skipped,
  # leaving an under-trained checkpoint in the results.
  if [ -f "runs/$NAME/.train_complete" ]; then
    echo "SKIP train $NAME - already completed (rm runs/$NAME/.train_complete to redo)"
    return 0
  fi
  banner "TRAIN $NAME started"
  mkdir -p "runs/$NAME"
  python -u -m pxai.train --config "configs/generated/$NAME.yaml" \
      > "runs/$NAME/train.log" 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    touch "runs/$NAME/.train_complete"
  else
    echo "TRAIN $NAME FAILED (exit $code) - no sentinel written, will retry next run"
  fi
  echo "===== TRAIN $NAME finished :: $(date) :: exit $code ====="
}

eval_one () {
  NAME=$1
  if [ ! -f "runs/$NAME/best.pt" ]; then
    echo "SKIP eval $NAME - no checkpoint (training must have failed)"
    return 0
  fi
  banner "EVAL $NAME started"
  python -u -m pxai.evaluate --config "configs/generated/$NAME.yaml" \
      --ckpt "runs/$NAME/best.pt" > "runs/$NAME/eval.log" 2>&1
  echo "===== EVAL $NAME finished :: $(date) :: exit $? ====="
}

echo "########## NIGHT RUN STARTED $(date) ##########"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null || true

# ---- 0. freeze the current known-good state before anything is overwritten ----
banner "SNAPSHOT current state"
./snapshot_stable.sh "pre_night_$(date +%Y%m%d)" || {
  echo "SNAPSHOT FAILED - aborting so nothing gets overwritten"; exit 1; }

# ---- 1. shuffle control on the FULL test set (decision-critical, shortest) ----
banner "SHUFFLE CONTROL started"
python -u probe_infidelity_shuffle.py \
    --config configs/generated/A2_protopnet_mobilevit_120ep.yaml \
    --ckpt   runs/A2_protopnet_mobilevit_120ep/best.pt \
    --n 128 --device $DEV \
    > runs/shuffle_control_full.log 2>&1
echo "===== SHUFFLE CONTROL finished :: $(date) :: exit $? ====="

# ---- 2. A2 ProtoPNet re-eval with all three infidelity variants ----
#         (no retrain: best.pt unchanged, only the metric list differs).
#         Early because it is cheap and yields the infidelity comparison table.
eval_one A2_protopnet_mobilevit_120ep

# ---- 3 + 4. ConvNeXt black box at 120 epochs, then evaluate ----
#             This is the budget-parity fix for the accuracy claim.
train_one ref_blackbox_convnext_120ep
eval_one  ref_blackbox_convnext_120ep

# ---- 5 + 6. B-cos head at 120 epochs, then evaluate (bonus if time allows) ----
train_one A2_bcos_mobilevit_120ep
eval_one  A2_bcos_mobilevit_120ep

echo ""
echo "########## ALL DONE $(date) ##########"