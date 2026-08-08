#!/usr/bin/env bash
# run_rrr.sh — does suppressing the shortcut close the localisation/faithfulness gap?
#
#   ./run_rrr.sh calibrate   # find a lambda that bites without dominating   ~15m
#   ./run_rrr.sh train       # 3 heads x chosen lambda                       ~5h
#   ./run_rrr.sh measure     # occlusion, localisation, faithfulness         ~4h
#   ./run_rrr.sh             # all three                                     ~9h
#
# THE QUESTION
# Part II SEC 5.5 measured a trade-off: swapping to the gradient read-out improves
# localisation 2-5x and degrades deletion 2-8x, on all three heads. The interpretation
# offered was that the two metrics only agree when the model depends on the object --
# and Part I showed this model does not, it depends on background.
#
# That is an interpretation. This tests it. Train a model that cannot use background
# and the two explanations should converge.
#
#   PREDICTIONS, RECORDED BEFORE THE RUN
#     deletion ratio (gradattr / native)   8.0x -> toward 1.0x
#     egg-masked accuracy                  0.2485 -> toward chance 0.0909
#     native conc_+                        2.53 -> up
#     test accuracy                        some loss expected; report it
#
#   If the ratio does not move, SEC 5.5.1 is wrong and the trade-off has another cause.
#   That is a real outcome, not a failed experiment.
#
# PRIOR WORK: Ross, Hughes & Doshi-Velez (2017), IJCAI. The loss is theirs. What is new
# is using it as an instrument on a control-validated shortcut and asking whether the
# explanation gap closes -- which needs both axes measured, and neither Ross nor GAIN
# reports both.
#
# NOTHING IS OVERWRITTEN: new run directories, new configs. Existing checkpoints and
# results untouched; they are the controls.

set -uo pipefail
cd "$(dirname "$0")"

LAMBDA="${RRR_LAMBDA:-1.0}"
HEADS=(protopnet cbm_sup bcos)
PAR=3
MASTER="logs/rrr_$(date +%Y%m%d_%H%M).log"
mkdir -p logs figs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

guard_idle() {
  local n; n=$(pgrep -fc "pxai.(train|evaluate)" || true)
  [ "${n:-0}" -gt 0 ] && { say "ABORT: $n pxai process(es) running"; return 1; }
  return 0
}

# --------------------------------------------------------------- stage: calibrate
stage_calibrate() {
  say "===== calibrate — find a lambda that bites without dominating ====="
  say "  Ross et al.: set lambda so the task and penalty terms are the same order of"
  say "  magnitude. Ratio far below 0.1 -> too weak. Far above 10 -> accuracy dies."
  [ -f pxai/rrr_penalty.py ] || cp rrr_penalty.py pxai/
  grep -q build_rrr pxai/train.py 2>/dev/null || \
    { python apply_rrr.py --check && python apply_rrr.py; } 2>&1 | tee -a "$MASTER"
  python -c "import pxai.train; print('  imports OK')" 2>&1 | tee -a "$MASTER"

  for lam in 0.1 1.0 10.0; do
    say "-- lambda $lam --"
    python - "$lam" <<'PY' 2>&1 | tee -a "$MASTER"
import sys, yaml, copy
lam = float(sys.argv[1])
c = yaml.safe_load(open("configs/generated/roi477_protopnet_120ep.yaml"))
c["train"]["rrr_lambda"] = lam
c["train"]["epochs"] = 1
c["output_dir"] = "./runs/_rrr_cal"
yaml.safe_dump(c, open("configs/generated/_rrr_cal.yaml", "w"), sort_keys=False)
PY
    timeout 600 python -u preflight_learns.py --config configs/generated/_rrr_cal.yaml \
        --device cuda --steps 60 >> "$MASTER" 2>&1 || true
    grep "\[rrr\]" "$MASTER" | tail -3
  done
  say "  pick the lambda whose ratio is nearest 1, then: RRR_LAMBDA=<it> ./run_rrr.sh train"
}

# ------------------------------------------------------------------ stage: train
stage_train() {
  guard_idle || return 1
  say "===== train — 3 heads, lambda=$LAMBDA (-P $PAR) ====="
  python - "$LAMBDA" <<'PY' 2>&1 | tee -a "$MASTER"
import sys, yaml, copy
lam = float(sys.argv[1])
for h in ("protopnet", "cbm_sup", "bcos"):
    src = f"configs/generated/roi477_{h}_120ep.yaml"
    c = yaml.safe_load(open(src))
    c["train"]["rrr_lambda"] = lam
    name = f"roi477_{h}_rrr_120ep"
    c["output_dir"] = f"./runs/{name}"
    yaml.safe_dump(c, open(f"configs/generated/{name}.yaml", "w"), sort_keys=False)
    print(f"  {name}  lambda={lam}")
PY
  printf '%s\n' "${HEADS[@]}" | sed 's/^/roi477_/; s/$/_rrr_120ep/' | \
    xargs -P "$PAR" -I{} bash -c '
      a="{}"
      [ -f "runs/$a/.train_complete" ] && { echo "  skip  $a"; exit 0; }
      mkdir -p "runs/$a"; echo "  start $a"
      python -u -m pxai.train --config "configs/generated/$a.yaml" \
        > "runs/$a/train.log" 2>&1 && touch "runs/$a/.train_complete" \
        && echo "  done  $a $(grep -o "test_acc=[0-9.]*" runs/$a/train.log | tail -1)" \
        || echo "  FAIL  $a"' 2>&1 | tee -a "$MASTER"
}

# ---------------------------------------------------------------- stage: measure
stage_measure() {
  guard_idle || return 1
  say "===== measure ====="
  ARMS=(); for h in "${HEADS[@]}"; do ARMS+=("roi477_${h}_rrr_120ep"); done

  say "-- did the shortcut shrink? (egg-masked; baseline protopnet 0.2485, chance 0.0909)"
  for a in "${ARMS[@]}"; do
    [ -f "runs/$a/best.pt" ] || continue
    python -u probe_occlusion_v2.py --config "configs/generated/$a.yaml" \
        --ckpt "runs/$a/best.pt" --labels ../Data/chula_roi2_w477/labels.json \
        --device cuda --emit-tsv runs/occlusion_sweep.tsv >> "$MASTER" 2>&1 || true
    echo "  occ $a"; tail -14 "$MASTER"
  done

  say "-- localisation, native vs gradient"
  printf '%s\n' "${ARMS[@]}" | xargs -P "$PAR" -I{} bash -c '
    a="{}"; [ -f "runs/$a/best.pt" ] || exit 0
    python -u probe_gradattr.py --device cuda --runs "$a" \
      --emit-tsv "figs/gradattr_$a.tsv" > "logs/ga_$a.log" 2>&1 \
      && echo "  loc $a" || echo "  FAIL loc $a"' 2>&1 | tee -a "$MASTER"

  say "-- faithfulness, native then gradient (the headline)"
  for a in "${ARMS[@]}"; do
    [ -f "runs/$a/best.pt" ] || continue
    python -u -m pxai.evaluate --config "configs/generated/$a.yaml" \
      --ckpt "runs/$a/best.pt" > "runs/$a/eval_native.log" 2>&1 || true
    [ -f "runs/$a/results.json" ] && cp "runs/$a/results.json" "runs/$a/results.json.native-attr"
    PXAI_GRADATTR=1 PXAI_GRADATTR_SMOOTH=1 python -u -m pxai.evaluate \
      --config "configs/generated/$a.yaml" --ckpt "runs/$a/best.pt" \
      > "runs/$a/eval_gradattr.log" 2>&1 || true
    echo "  faith $a"
  done 2>&1 | tee -a "$MASTER"

  python - <<'PY' 2>&1 | tee -a "$MASTER"
import json, os
BASE = {"protopnet": (0.0166, 0.1333), "cbm_sup": (0.0290, 0.1440),
        "bcos": (0.0465, 0.0963)}
print(f"\n  DELETION RATIO  gradattr / native   (baseline -> RRR)")
print(f"  {'head':<12}{'baseline':>10}{'rrr':>10}{'change':>26}")
print("  " + "-" * 58)
for h, (bn, bg) in BASE.items():
    a = f"roi477_{h}_rrr_120ep"
    n, g = f"runs/{a}/results.json.native-attr", f"runs/{a}/results.json"
    if not (os.path.exists(n) and os.path.exists(g)):
        print(f"  {h:<12}{bg/bn:>9.1f}x{'--':>10}"); continue
    dn, dg = json.load(open(n)), json.load(open(g))
    key = next((k for k in dn.get("methods", {}) if k.startswith("ours:")), None)
    if not key: continue
    x = dn["methods"][key]["faithfulness"].get("deletion", float("nan"))
    y = dg.get("methods", {}).get(key, {}).get("faithfulness", {}).get("deletion", float("nan"))
    r = y / x if x else float("nan")
    verd = "GAP CLOSED" if r < 2.0 else ("narrowed" if r < bg/bn else "unchanged/wider")
    print(f"  {h:<12}{bg/bn:>9.1f}x{r:>9.1f}x{verd:>26}")
print("""
  ratio -> 1.0  ->  SEC 5.5.1 CONFIRMED: the trade-off WAS the shortcut. Suppress the
                    shortcut and the two explanations agree. Strong result.
  ratio unmoved ->  SEC 5.5.1 is wrong; the trade-off has another cause. Also a result,
                    and it means the SEC 5.5.3 granularity control becomes the priority.""")
PY
}

case "${1:-all}" in
  calibrate) stage_calibrate ;;
  train)     stage_train ;;
  measure)   stage_measure ;;
  all)       stage_calibrate; stage_train; stage_measure ;;
  *)         echo "usage: $0 [all|calibrate|train|measure]"; exit 2 ;;
esac
say "done. master log: $MASTER"
