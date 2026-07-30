#!/bin/bash

run_one () {
  NAME=$1
  if [ ! -f runs/$NAME/best.pt ]; then
    echo "SKIP $NAME - no checkpoint found"
    return
  fi
  echo "=== $NAME started at $(date) ==="
  mkdir -p runs/$NAME
  python -u -m pxai.evaluate \
    --config configs/generated/$NAME.yaml \
    --ckpt runs/$NAME/best.pt \
    > runs/$NAME/eval.log 2>&1
  CODE=$?
  echo "=== $NAME finished at $(date), exit code $CODE ==="
}

run_one ref_blackbox_convnext
run_one A2_protopnet_mobilevit

echo "ALL DONE at $(date)"
