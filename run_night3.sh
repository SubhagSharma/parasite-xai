#!/bin/bash
# run_night3.sh — cropped-data experiments. STRICTLY SERIAL.
#
# PHASE 1  train crop_protopnet   ~4h
# PHASE 2  train crop_blackbox    ~4h
# PHASE 3  train crop_bcos        ~4h
# PHASE 4  accuracy, all three    ~3m
# PHASE 5  localisation probes on crop ProtoPNet   ~30m   <- THE DECISIVE TEST
#          did cropping move the prototypes onto the eggs?
# PHASE 6  OOD flagging (IPI-CVx step 1)          ~10m
#          flags over/under-confident train samples; the ADD-BACK vs
#          REMOVE decision is deliberately left for tomorrow
# PHASE 7  full faithfulness eval, crop ProtoPNet ~3h
#                                 ~15.7h total
#
# Nothing existing is overwritten: all three write to NEW run directories, and a
# full snapshot of runs/ and configs/ is taken first.
#
#   nohup ./run_night3.sh > runs/night3.log 2>&1 &

set -u
cd "$(dirname "$0")"
mkdir -p runs
banner () { echo ""; echo "===== $1 :: $(date) ====="; }

LABELS=../Data/Chula-ParasiteEgg-11/labels.json

train_one () {
  NAME=$1
  # guard on a completion sentinel, not best.pt: a run killed part-way leaves a
  # best.pt behind that would otherwise look finished and be skipped.
  if [ -f "runs/$NAME/.train_complete" ]; then
    echo "SKIP train $NAME - already completed (rm runs/$NAME/.train_complete to redo)"
    return 0
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
  python -u eval_accuracy_only.py --config "configs/generated/$NAME.yaml" \
      --ckpt "runs/$NAME/best.pt" 2>&1 | tee -a "runs/$NAME/accuracy.log"
}

echo "########## NIGHT 3 (cropped data, serial) STARTED $(date) ##########"
banner "SNAPSHOT"
./snapshot_stable.sh "pre_night3_$(date +%Y%m%d_%H%M)" || { echo "SNAPSHOT FAILED"; exit 1; }

train_one crop_protopnet_120ep
train_one crop_blackbox_120ep
train_one crop_bcos_120ep

banner "ACCURACY"
acc_one crop_protopnet_120ep
acc_one crop_blackbox_120ep
acc_one crop_bcos_120ep

banner "LOCALISATION PROBES (the decisive test)"
if [ -f runs/crop_protopnet_120ep/best.pt ]; then
  echo "--- prototype placement: are the prototypes on the egg now? ---"
  python -u probe_prototype_localisation.py \
      --config configs/generated/crop_protopnet_120ep.yaml \
      --ckpt   runs/crop_protopnet_120ep/best.pt \
      --labels $LABELS --device cpu 2>&1 | tail -30
  echo ""
  echo "--- pointing game ---"
  python -u probe_pointing_game.py \
      --config configs/generated/crop_protopnet_120ep.yaml \
      --ckpt   runs/crop_protopnet_120ep/best.pt \
      --labels $LABELS --n 256 --device cpu --methods ante,gradcam 2>&1 | tail -15
else
  echo "SKIP probes - crop_protopnet has no checkpoint"
fi

banner "OOD FLAGGING (IPI-CVx step 1) on crop ProtoPNet"
if [ -f runs/crop_protopnet_120ep/best.pt ]; then
  python -u probe_ood_flagging.py \
      --config configs/generated/crop_protopnet_120ep.yaml \
      --ckpt   runs/crop_protopnet_120ep/best.pt \
      --tau 0.7 --k 0.02 --device cuda 2>&1 | tail -25
else
  echo "SKIP OOD flagging - no checkpoint"
fi

banner "FULL FAITHFULNESS EVAL: crop ProtoPNet"
if [ -f runs/crop_protopnet_120ep/best.pt ]; then
  python -u -m pxai.evaluate --config configs/generated/crop_protopnet_120ep.yaml \
      --ckpt runs/crop_protopnet_120ep/best.pt \
      > runs/crop_protopnet_120ep/eval.log 2>&1
  echo "eval exit $?"
fi

echo ""
banner "ACCURACY TABLE"
cat runs/accuracy_2x2.tsv 2>/dev/null || echo "(none)"
echo ""
echo "########## ALL DONE $(date) ##########"
