#!/bin/bash
# run_night2.sh — fill the head x backbone 2x2. STRICTLY SERIAL.
#
#   1. snapshot current state
#   2. TRAIN mobilevit_xs + blackbox   ~4h   <- the decisive control, cheapest
#   3. accuracy check                  ~1m
#   4. TRAIN convnext_tiny + protopnet ~8h   <- completes the 2x2
#   5. accuracy check                  ~1m
#                                     ~12h
#
# Only ACCURACY is computed (eval_accuracy_only.py). The faithfulness sweep is not
# needed to answer the head-vs-backbone question and would cost hours per model.
#
#   nohup ./run_night2.sh > runs/night2.log 2>&1 &

set -u
cd "$(dirname "$0")"
mkdir -p runs

banner () { echo ""; echo "===== $1 :: $(date) ====="; }

train_one () {
  NAME=$1
  if [ -f "runs/$NAME/.train_complete" ]; then
    echo "SKIP train $NAME - already completed"; return 0
  fi
  banner "TRAIN $NAME started"
  mkdir -p "runs/$NAME"
  python -u -m pxai.train --config "configs/generated/$NAME.yaml" \
      > "runs/$NAME/train.log" 2>&1
  code=$?
  [ $code -eq 0 ] && touch "runs/$NAME/.train_complete" \
                  || echo "TRAIN $NAME FAILED (exit $code) - will retry next run"
  echo "===== TRAIN $NAME finished :: $(date) :: exit $code ====="
}

acc_one () {
  NAME=$1
  [ -f "runs/$NAME/best.pt" ] || { echo "SKIP acc $NAME - no checkpoint"; return 0; }
  banner "ACCURACY $NAME"
  python -u eval_accuracy_only.py --config "configs/generated/$NAME.yaml" \
      --ckpt "runs/$NAME/best.pt" 2>&1 | tee -a "runs/$NAME/accuracy.log"
}

echo "########## NIGHT 2 STARTED $(date) ##########"

banner "SNAPSHOT"
./snapshot_stable.sh "pre_night2_$(date +%Y%m%d)" || { echo "SNAPSHOT FAILED"; exit 1; }

# decisive control first: same backbone as our interpretable models, linear head
train_one blackbox_mobilevit_120ep
acc_one   blackbox_mobilevit_120ep

# completes the 2x2: same backbone as the black box, interpretable head
train_one A1_convnext_tiny_120ep
acc_one   A1_convnext_tiny_120ep

echo ""
banner "2x2 TABLE SO FAR"
cat runs/accuracy_2x2.tsv 2>/dev/null || echo "(none)"
echo ""
echo "########## ALL DONE $(date) ##########"
