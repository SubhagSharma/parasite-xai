#!/usr/bin/env bash
# run_sea_night3.sh — the full night, one command.
#
#   nohup ./run_sea_night3.sh > logs/night3_wrapper.log 2>&1 &
#
# Stages, in dependency order. Sentinel-guarded: a kill or reboot resumes.
#   train   4 arms (whole/crop x s32/s4)          -P 3   ~5.5 h
#   eval    8 SEA arms                            -P 2   ~1.5 h
#   noise   KernelSHAP self-consistency ceiling   -P 1   ~0.4 h
#   report  batch_visualise + eps rescore         -P 1   ~1.3 h
#                                                 TOTAL  ~8.7 h
#
# The two questions this night answers:
#   1. Does the s32 -> s4 gain SHRINK as the egg gets bigger (whole 1.08 cells
#      -> roi477 2.40 -> crop 5.17)? That monotone shrink is the mechanism
#      claim. A large gain on crop falsifies it.
#   2. Does KernelSHAP agree with itself above 0.70? If not, C2's gate is
#      unreachable and a month of FastSHAP work is unjustified.
set -euo pipefail
cd "$(dirname "$0")"

TRAIN=(whole_sea_s32 whole_sea_s4 crop_sea_s32 crop_sea_s4)
EVAL=(roi477_sea_s32 roi477_sea_s16 roi477_sea_s8 roi477_sea_s4
      whole_sea_s32 whole_sea_s4 crop_sea_s32 crop_sea_s4)
NOISE_RUNS=roi477_bcos_120ep,roi477_sea_s8
MASTER="logs/night3_$(date +%Y%m%d_%H%M).log"
mkdir -p logs figs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

stage_train() {
  say "=== train (${#TRAIN[@]} arms, -P 3) ==="
  for a in "${TRAIN[@]}"; do
    [ -f "configs/generated/$a.yaml" ] || { say "ABORT: $a.yaml missing"; exit 1; }
    mkdir -p "runs/$a"
  done
  ./snapshot_stable.sh pre-night3 >>"$MASTER" 2>&1 || say "WARN: snapshot failed"
  printf '%s\n' "${TRAIN[@]}" | xargs -P 3 -I{} bash -c '
    a="{}"; [ -f "runs/$a/.train_complete" ] && { echo "  skip  $a"; exit 0; }
    echo "  start $a"
    if python -u -m pxai.train --config "configs/generated/$a.yaml" \
         > "runs/$a/train.log" 2>&1; then
      touch "runs/$a/.train_complete"; echo "  done  $a $(tail -1 runs/$a/train.log)"
    else echo "  FAIL  $a -- runs/$a/train.log"; fi' 2>&1 | tee -a "$MASTER"
}

stage_eval() {
  say "=== eval (${#EVAL[@]} arms, -P 2) ==="
  printf '%s\n' "${EVAL[@]}" | xargs -P 2 -I{} bash -c '
    a="{}"
    [ -f "runs/$a/best.pt" ] || { echo "  skip  $a (no ckpt)"; exit 0; }
    [ -f "runs/$a/.eval_complete" ] && { echo "  skip  $a"; exit 0; }
    [ -f "runs/$a/results.json" ] && cp "runs/$a/results.json" "runs/$a/results.json.pre-night3"
    echo "  start $a"
    if python -u -m pxai.evaluate --config "configs/generated/$a.yaml" \
         --ckpt "runs/$a/best.pt" > "runs/$a/eval.log" 2>&1; then
      touch "runs/$a/.eval_complete"; echo "  done  $a"
    else echo "  FAIL  $a -- runs/$a/eval.log"; fi' 2>&1 | tee -a "$MASTER"
}

stage_noise() {
  say "=== noise — KernelSHAP self-consistency ceiling ==="
  python -u probe_kernelshap_noise.py --device cuda --runs "$NOISE_RUNS" \
    --n-images 20 --out figs/kernelshap_noise.json 2>&1 | tee -a "$MASTER" \
    || say "WARN: noise probe failed"
}

stage_report() {
  say "=== report ==="
  cp figs/attribution_metrics.tsv figs/attribution_metrics.tsv.pre-night3 2>/dev/null || true
  # serial on purpose: concurrent appends corrupt the TSV (HANDOFF sec 7)
  python -u batch_visualise.py --runs '*_sea_s*' --no-figs \
    --tsv figs/attribution_metrics.tsv --device cuda 2>&1 | tail -15 | tee -a "$MASTER"
  python add_loc_efficiency.py --write 2>&1 | tee -a "$MASTER"
  say "  --- SEA dose-response: eps by dataset x stride ---"
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import pandas as pd
try: d = pd.read_csv("figs/attribution_metrics.eps.tsv", sep="\t")
except Exception as e: print("  ", e); raise SystemExit
s = d[d.run.str.contains("_sea_s", na=False)].copy()
if not len(s): print("  no SEA rows"); raise SystemExit
s["stride"] = s.run.str.extract(r"_sea_s(\d+)")[0].astype(int)
s["ds"] = s.run.str.split("_sea_").str[0]
print(s.pivot_table(index=["ds","stride"], columns="method",
                    values="eps_pos", aggfunc="mean").round(3).to_string())
PY
}

case "${1:-all}" in
  train) stage_train ;; eval) stage_eval ;; noise) stage_noise ;;
  report) stage_report ;;
  all) stage_train; stage_eval; stage_noise; stage_report ;;
  *) echo "usage: $0 [all|train|eval|noise|report]"; exit 2 ;;
esac
say "done. master log: $MASTER"