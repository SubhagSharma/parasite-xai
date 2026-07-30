#!/bin/bash
# run_aug_sweep.sh — augmentation sweep, GPU-saturating by default.
#
# The four trainings are independent (separate output dirs, separate logs) so they can
# run CONCURRENTLY. One job used ~2.9GB of the 20GB slice and left the GPU at ~14%
# utilisation, so four together still fit with margin and keep the device busy.
#
#   PAR=4 ./run_aug_sweep.sh     4 at once (default, fastest)
#   PAR=1 ./run_aug_sweep.sh     strictly serial
#
# batch_size is untouched on purpose: aug0_current is a control that must reproduce
# the existing crop model's egg-masked 0.2494, which trained at batch 32.
#
#   nohup ./run_aug_sweep.sh > runs/aug_sweep.log 2>&1 &

set -u
cd "$(dirname "$0")"
mkdir -p runs
LABELS=../Data/Chula-ParasiteEgg-11/labels.json
HEAD=${HEAD:-protopnet}
PAR=${PAR:-4}
LEVELS="aug0_current aug1_hue aug2_strong aug3_gray"

banner () { echo ""; echo "===== $1 :: $(date) ====="; }

echo "########## AUG SWEEP STARTED $(date)  (PAR=$PAR) ##########"
nproc 2>/dev/null | sed 's/^/host CPUs: /' || true
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null || true
./snapshot_stable.sh "pre_augsweep_$(date +%Y%m%d_%H%M)" || { echo "SNAPSHOT FAILED"; exit 1; }

# ---------------- phase 1: train, up to PAR at a time ----------------
banner "PHASE 1: training (${PAR} concurrent)"
running=0
for L in $LEVELS; do
  NAME="crop_${HEAD}_${L}"
  if [ -f "runs/$NAME/.train_complete" ]; then
    echo "SKIP $NAME - already complete"; continue
  fi
  mkdir -p "runs/$NAME"
  ( python -u -m pxai.train --config "configs/generated/$NAME.yaml" \
        > "runs/$NAME/train.log" 2>&1
    c=$?
    [ $c -eq 0 ] && touch "runs/$NAME/.train_complete"
    echo "  [$NAME] train exit $c :: $(date)" ) &
  echo "  launched $NAME (pid $!)"
  running=$((running+1))
  if [ "$running" -ge "$PAR" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
done
wait
banner "PHASE 1 complete"
nvidia-smi --query-gpu=memory.used --format=csv 2>/dev/null || true

# ---------------- phase 2: measure the shortcut (CPU) ----------------
banner "PHASE 2: accuracy + shortcut per level"
for L in $LEVELS; do
  NAME="crop_${HEAD}_${L}"
  [ -f "runs/$NAME/best.pt" ] || { echo "SKIP $NAME - no checkpoint"; continue; }
  echo ""
  echo "--- $NAME ---"
  python -u eval_accuracy_only.py --config "configs/generated/$NAME.yaml" \
      --ckpt "runs/$NAME/best.pt" 2>&1 | tail -4
  python -u probe_occlusion.py --config "configs/generated/$NAME.yaml" \
      --ckpt "runs/$NAME/best.pt" --labels $LABELS --device cpu 2>&1 \
      | grep -E "unmodified|EGG masked|BACKGROUND masked|chance level"
done

banner "SUMMARY"
echo "chance = 0.0909   current crop model egg-masked = 0.2494   lower is better"
printf "%-30s %10s %12s\n" "level" "accuracy" "egg-masked"
for L in $LEVELS; do
  NAME="crop_${HEAD}_${L}"
  [ -f "runs/$NAME/best.pt" ] || continue
  OUT=$(python -u probe_occlusion.py --config "configs/generated/$NAME.yaml" \
        --ckpt "runs/$NAME/best.pt" --labels $LABELS --device cpu 2>/dev/null)
  A=$(echo "$OUT" | grep "unmodified"   | grep -oE "[0-9]\.[0-9]+")
  E=$(echo "$OUT" | grep "EGG masked"   | grep -oE "[0-9]\.[0-9]+")
  printf "%-30s %10s %12s\n" "$L" "${A:-?}" "${E:-?}"
done
echo ""
echo "########## ALL DONE $(date) ##########"
