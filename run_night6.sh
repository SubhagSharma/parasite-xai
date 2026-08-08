#!/bin/bash
# run_night6.sh — supervised CBM, seed-2337 replication, faithfulness on patched code.
#
# Ordered so the CHEAPEST high-value result lands first. If the box dies at 04:00 you
# keep the seed replication, which is what upgrades the project's headline claim.
#
# PHASE 0  snapshot + preflight (smoke test, ConvNeXt probe)      ~25m
# PHASE 1  train roi477_cbm_sup   (supervised 23-d bottleneck)    ~0.6h
# PHASE 2  SEED 2337 x 4 heads                                    ~3.7h
# PHASE 3  accuracy + occlusion + displaced control, seed set     ~0.4h
#            <- head ordering PROVISIONAL -> ESTABLISHED here
# PHASE 4  faithfulness x 4 on the patched code                   ~13.3h
#                                        GUARANTEED TOTAL ~18.3h
# PHASE 5  OVERFLOW: whole-image faithfulness x 3                 +10h
#
# THREE CODE CHANGES SINCE NIGHT 5 — every faithfulness number below is on new code:
#   * |rho| per sample in sanity_check (signed values were cancelling; B-cos's true
#     dependence was 2.5x its reported figure), 3-seed averaging, collapse rate
#   * sanity_check excluded from normalised_aggregates (three incomparable regimes)
#   * CBM attribution now class-conditional through the bottleneck; it was a
#     class-agnostic backbone channel mean that never touched the head
#
#   chmod +x run_night6.sh
#   nohup ./run_night6.sh > runs/night6.log 2>&1 &

set -u
cd "$(dirname "$0")"
mkdir -p runs
banner () { echo ""; echo "===== $1 :: $(date) ====="; }

DATA=../Data
ROI=$DATA/chula_roi2_w477
LAB=$ROI/labels.json
SEED=2337
SEEDSET="roi477_blackbox_s${SEED}_120ep roi477_protopnet_s${SEED}_120ep \
roi477_bcos_s${SEED}_120ep roi477_cbm_sup_s${SEED}_120ep"
FAITH="roi477_protopnet_120ep roi477_blackbox_120ep roi477_bcos_120ep roi477_cbm_sup_120ep"
RUN_OVERFLOW=1

find pxai -path "*__pycache__*" -name "*.pyc" -delete 2>/dev/null

train_one () {
  N=$1
  [ -f "configs/generated/$N.yaml" ] || { echo "SKIP $N - NO CONFIG"; return 0; }
  [ -f "runs/$N/.train_complete" ] && { echo "SKIP $N - done"; return 0; }
  banner "TRAIN $N"
  mkdir -p "runs/$N"
  python -u -m pxai.train --config "configs/generated/$N.yaml" > "runs/$N/train.log" 2>&1
  c=$?
  [ $c -eq 0 ] && touch "runs/$N/.train_complete" || echo "TRAIN $N FAILED ($c)"
  echo "===== $N :: $(date) :: exit $c ====="
  # the concept report is the point of the supervised run; surface it
  case "$N" in *cbm_sup*) grep -A 30 "MACRO" "runs/$N/train.log" | tail -34 ;; esac
}

probe_one () {
  N=$1
  [ -f "runs/$N/best.pt" ] || { echo "SKIP probes $N - no checkpoint"; return 0; }
  python -u eval_accuracy_only.py --config "configs/generated/$N.yaml" \
      --ckpt "runs/$N/best.pt" 2>&1 | tee -a "runs/$N/accuracy.log"
  python -u probe_occlusion_v2.py --config "configs/generated/$N.yaml" \
      --ckpt "runs/$N/best.pt" --labels "$LAB" --device cuda \
      --emit-tsv runs/occlusion_sweep.tsv 2>&1 | tail -14
  python -u probe_displaced_control.py --config "configs/generated/$N.yaml" \
      --ckpt "runs/$N/best.pt" --labels "$LAB" --device cuda \
      --emit-tsv runs/displaced_control.tsv 2>&1 | tail -12
}

evaluate_one () {
  N=$1
  [ -f "runs/$N/best.pt" ] || { echo "SKIP eval $N - no checkpoint"; return 0; }
  [ -f "runs/$N/.eval_complete" ] && { echo "SKIP eval $N - done"; return 0; }
  [ -f "runs/$N/results.json" ] && \
      cp "runs/$N/results.json" "runs/$N/results.json.pre-night6"
  banner "FAITHFULNESS $N"
  python -u -m pxai.evaluate --config "configs/generated/$N.yaml" \
      --ckpt "runs/$N/best.pt" > "runs/$N/eval_v2.log" 2>&1
  c=$?
  [ $c -eq 0 ] && touch "runs/$N/.eval_complete" || echo "EVAL $N FAILED ($c)"
  echo "===== eval $N :: $(date) :: exit $c ====="
  grep -E "sanity_collapse_rate|sanity_seed_std|aggregate" "runs/$N/results.json" 2>/dev/null | head -8
}

echo "########## NIGHT 6 STARTED $(date) ##########"

banner "PHASE 0a  snapshot"
./snapshot_stable.sh "pre_night6_$(date +%Y%m%d_%H%M)" || { echo "SNAPSHOT FAILED"; exit 1; }
[ -f "$LAB" ] || { echo "MISSING $LAB"; exit 1; }
[ -f "$DATA/Chula-ParasiteEgg-11/concepts_v3.csv" ] || { echo "MISSING concepts_v3.csv"; exit 1; }

banner "PHASE 0b  configs"
python -u make_night6_configs.py

banner "PHASE 0c  patch verification"
python - <<'PY'
import pxai.eval.faithfulness as f, inspect
import pxai.evaluate as ev
print("  AGGREGATE_EXCLUDE :", sorted(f.AGGREGATE_EXCLUDE))
print("  SANITY_SEEDS      :", getattr(f, "SANITY_SEEDS", "MISSING"))
src = inspect.getsource(ev.ante_hoc_attr)
print("  cbm attribution   :",
      "PATCHED" if "classifier.weight[target]" in src else "** STILL THE OLD MEAN **")
PY

banner "PHASE 0d  ConvNeXt x ProtoPNet saturation probe"
python -u preflight_learns.py --config configs/generated/A1_convnext_tiny_120ep.yaml \
    --device cuda --steps 300 || echo "(expected FAIL — this is the diagnosis)"

banner "PHASE 0e  faithfulness smoke test (1 batch)"
SMOKE=configs/generated/_smoke_n6.yaml
sed -e 's/^  faithfulness_batches:.*/  faithfulness_batches: 1/' \
    -e 's/^    sensitivity:.*/    sensitivity: 1/' \
    -e 's/^    sanity_check:.*/    sanity_check: 1/' \
    -e 's|^output_dir:.*|output_dir: ./runs/_smoke_n6|' \
    configs/generated/roi477_protopnet_120ep.yaml > "$SMOKE"
mkdir -p runs/_smoke_n6
python -u -m pxai.evaluate --config "$SMOKE" \
    --ckpt runs/roi477_protopnet_120ep/best.pt > runs/_smoke_n6/eval.log 2>&1
SMOKE_OK=$?
echo "smoke exit $SMOKE_OK"
[ $SMOKE_OK -ne 0 ] && tail -25 runs/_smoke_n6/eval.log

banner "PHASE 1  supervised CBM — 23 morphology concepts"
train_one roi477_cbm_sup_120ep
probe_one roi477_cbm_sup_120ep

banner "PHASE 2  SEED $SEED replication, 4 heads"
for N in $SEEDSET; do train_one "$N"; done

banner "PHASE 3  seed-set probes — the head ordering test"
echo "seed 1337 reference, egg-masked (box mask):"
echo "  protopnet 0.2485  cbm 0.3084  blackbox 0.3670  bcos 0.4128"
echo "If seed $SEED reproduces this ORDER, F6/F7 go PROVISIONAL -> ESTABLISHED."
for N in $SEEDSET; do probe_one "$N"; done

if [ $SMOKE_OK -eq 0 ]; then
  banner "PHASE 4  faithfulness on patched code, 4 heads"
  for N in $FAITH; do evaluate_one "$N"; done
else
  banner "PHASE 4 SKIPPED — smoke test failed, see runs/_smoke_n6/eval.log"
fi

echo ""
echo "########## GUARANTEED SCOPE COMPLETE $(date) ##########"

if [ "$RUN_OVERFLOW" = "1" ] && [ $SMOKE_OK -eq 0 ]; then
  banner "PHASE 5  OVERFLOW: whole-image faithfulness"
  for N in blackbox_mobilevit_120ep A2_protopnet_mobilevit_120ep A2_bcos_mobilevit_120ep; do
    evaluate_one "$N"
  done
fi

banner "SUMMARY"
echo "--- accuracy ---"
grep -E "roi477|run" runs/accuracy_2x2.tsv 2>/dev/null | tail -20
echo ""
echo "--- displaced control (egg-specific fraction) ---"
column -t runs/displaced_control.tsv 2>/dev/null | tail -22
echo ""
echo "########## ALL DONE $(date) ##########"
