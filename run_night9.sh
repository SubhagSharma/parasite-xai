#!/usr/bin/env bash
# run_night9.sh — everything, ordered so a truncated night still lands the important bits.
#
#   ./run_night9.sh                # all of it   (~17.5 h)
#   ./run_night9.sh preflight      # gates only  (~35 m)  <- RUN THIS FIRST
#   ./run_night9.sh letterbox | multiscale | cheap | faith | bcosnet
#
# PHASE 0  preflight: every gate, before any long run           ~35m
# PHASE 1  letterbox anisotropy control                          ~4.5h
# PHASE 2  multi-scale prototypes, 4 arms                        ~2.5h
# PHASE 3  cheap fixes: audit, bcos=hirescam, push drift         ~40m
# PHASE 4  faithfulness under gradient attribution               ~6h
# PHASE 5  real B-cos network, 3 seeds                           ~4h
#                                                        TOTAL  ~17.7h
#
# ORDERING
# Phase 1 first because it is the only item that changes a number ALREADY IN THE
# REPORTS: the whole-image baseline carries an uncorrected 4.84x anisotropy, which
# inflates the apparent shortcut in the arm the ROI is compared against, making the
# reported "45-69% of the shortcut removed" an UPPER BOUND (report SEC 5.5.2).
# Phase 2 next: cheapest novel result, and it answers the cilia/plug question.
# Phase 4 late: it answers a Part II question, and Part II is already a replication.
# Phase 5 last: the B-cos prediction is a control, valuable but not load-bearing.
#
# GPU
# -P 3 for training (~4 GB each, ~85% util; single-process these sit near 20% because
# they are launch-latency bound). -P 2 for evaluation, which is heavier. Every stage
# calls guard_idle first, so nothing stacks accidentally.
#
# NOTHING IS OVERWRITTEN
# Every arm writes to a NEW run directory. Faithfulness copies results.json to
# .native-attr before touching it. Existing checkpoints, results and TSV rows survive.
#
#   chmod +x run_night9.sh
#   ./run_night9.sh preflight              # read the output before committing
#   nohup ./run_night9.sh > logs/night9_console.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")"

PAR_TRAIN=3
PAR_EVAL=2
MASTER="logs/night9_$(date +%Y%m%d_%H%M).log"
MS_ARMS=(roi477_ms_coarse_120ep roi477_ms_fine_120ep roi477_ms_2way_120ep roi477_ms_3way_120ep)
FAITH=(roi477_protopnet_120ep roi477_cbm_sup_120ep roi477_bcos_120ep)

mkdir -p logs figs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

guard_idle() {
  local n; n=$(pgrep -fc "pxai.(train|evaluate)" || true)
  if [ "${n:-0}" -gt 0 ]; then
    say "WAIT: $n pxai process(es) running; sleeping 5m"
    sleep 300
    n=$(pgrep -fc "pxai.(train|evaluate)" || true)
    [ "${n:-0}" -gt 0 ] && { say "ABORT: still busy"; return 1; }
  fi
  return 0
}

train_list() {
  local par="$1"; shift
  printf '%s\n' "$@" | xargs -P "$par" -I{} bash -c '
    a="{}"
    [ -f "configs/generated/$a.yaml" ] || { echo "  skip  $a (no config)"; exit 0; }
    [ -f "runs/$a/.train_complete" ] && { echo "  skip  $a (done)"; exit 0; }
    mkdir -p "runs/$a"; echo "  start $a"
    if python -u -m pxai.train --config "configs/generated/$a.yaml" \
         > "runs/$a/train.log" 2>&1; then
      touch "runs/$a/.train_complete"
      echo "  done  $a  $(grep -o "test_acc=[0-9.]*" runs/$a/train.log | tail -1)"
    else
      echo "  FAIL  $a -- runs/$a/train.log"
    fi' 2>&1 | tee -a "$MASTER"
}

probe_list() {
  local par="$1"; shift
  printf '%s\n' "$@" | xargs -P "$par" -I{} bash -c '
    a="{}"
    [ -f "runs/$a/best.pt" ] || exit 0
    python -u eval_accuracy_only.py --config "configs/generated/$a.yaml" \
        --ckpt "runs/$a/best.pt" >> "runs/$a/accuracy.log" 2>&1
    python -u probe_gradattr.py --device cuda --runs "$a" \
        --emit-tsv "figs/gradattr_$a.tsv" > "logs/ga_$a.log" 2>&1 \
      && echo "  probe $a" || echo "  FAIL probe $a"' 2>&1 | tee -a "$MASTER"
}

# =============================================================== PHASE 0 preflight
stage_preflight() {
  say "===== PHASE 0  preflight — every gate before any long run ====="
  local ok=1

  say "-- letterbox --"
  [ -f pxai/letterbox.py ] || python make_letterbox_config.py 2>&1 | tee -a "$MASTER"
  if ! grep -q "LetterboxSquare" pxai/data.py 2>/dev/null; then
    python apply_letterbox.py --check 2>&1 | tee -a "$MASTER"
    python apply_letterbox.py 2>&1 | tee -a "$MASTER"
  fi
  grep -c "letterbox" pxai/data.py | xargs -I{} say "  'letterbox' appears {} times in data.py (want >=4)"
  python - <<'PY' 2>&1 | tee -a "$MASTER"
try:
    from pxai.utils import load_config
    from pxai.data import build_loaders
    c = load_config('configs/generated/whole_blackbox_lb_120ep.yaml'); c['device'] = 'cpu'
    t = build_loaders(c).test.dataset
    t = getattr(t, 'dataset', t)
    names = [type(x).__name__ for x in t.transform.transforms]
    print(f"  transform chain: {names}")
    print("  LETTERBOX ACTIVE" if names and names[0] == 'LetterboxSquare'
          else "  ** LetterboxSquare NOT FIRST -- flag not threaded, do not train")
except Exception as e:
    print(f"  letterbox check FAILED: {type(e).__name__}: {e}")
PY

  say "-- multi-scale --"
  [ -f pxai/models/protopnet_multiscale.py ] || cp protopnet_multiscale.py pxai/models/
  if ! grep -q "MultiScaleProtoHead" pxai/models/__init__.py 2>/dev/null; then
    python apply_multiscale_head.py --check 2>&1 | tee -a "$MASTER"
    python apply_multiscale_head.py 2>&1 | tee -a "$MASTER"
  fi
  python -c "
import timm
m = timm.create_model('mobilevit_xs', pretrained=False, features_only=True)
print('  stage channels ', m.feature_info.channels())
print('  stage strides  ', m.feature_info.reduction())" 2>&1 | tee -a "$MASTER"
  # measured strides on this build are [2,4,8,16,32], so stage 4 (not 3) is the
  # stride-32 stage the baseline uses. With the defaults ms_coarse would sit at
  # stride 16 and would NOT be a wiring control. Stage 0 (stride 2) is a 112x112
  # grid -- far too slow -- so the 3-way arm uses stage 1 instead.
  # forward() is a SEPARATE patch: apply_multiscale_head.py does not touch it, and
  # without it every EVALUATION path hands the head one tensor instead of the pyramid.
  grep -q "pyramid_forward(self.backbone" pxai/models/__init__.py 2>/dev/null || \
    python apply_ms_forward_fix.py >> "$MASTER" 2>&1 || true
  python - <<'PYFWD' >> "$MASTER" 2>&1 || true
import torch
from pxai.utils import load_config
from pxai.data import build_loaders
from pxai.models import build_model
c = load_config('configs/generated/roi477_ms_2way_120ep.yaml'); c['device'] = 'cpu'
l = build_loaders(c); c['model']['num_classes'] = len(l.classes)
m = build_model(c).eval()
with torch.no_grad():
    o = m(torch.randn(4, 3, 224, 224))
print(f"  ms forward -> {tuple(o.shape)}  " +
      ("MS FORWARD OK" if tuple(o.shape) == (4, len(l.classes))
       else "** WRONG SHAPE -- do not run phase 2"))
PYFWD
  tail -3 "$MASTER"

  python make_multiscale_configs.py --fine 1 --mid 2 --coarse 4 >> "$MASTER" 2>&1 || true; tail -8 "$MASTER"
  python -u preflight_learns.py --config configs/generated/roi477_ms_2way_120ep.yaml \
      --device cuda >> "$MASTER" 2>&1 || true; tail -6 "$MASTER"

  say "-- gradattr eval flag --"
  grep -q "PXAI_GRADATTR" pxai/evaluate.py 2>/dev/null || \
    python apply_gradattr_eval.py 2>&1 | tee -a "$MASTER"
  PXAI_GRADATTR=1 python -u batch_visualise.py --runs roi477_protopnet_120ep --fast \
      --outdir /tmp/gchk --tsv /tmp/gchk.tsv --device cuda >/dev/null 2>&1
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import csv, statistics as st, os
p = "/tmp/gchk.tsv"
v = [float(r["conc_pos"]) for r in csv.DictReader(open(p), delimiter="\t")
     if r["method"] == "ours:protopnet" and r["conc_pos"] not in ("nan", "")] \
    if os.path.exists(p) else []
if v:
    m = st.mean(v)
    print(f"  conc_pos {m:.2f} -> " +
          ("GRADATTR ACTIVE" if m > 4.0 else "** FLAG NOT BITING -- skip phase 4"))
else:
    print("  ** no output -- skip phase 4")
PY

  say "-- bcos network --"
  python -c "import bcos; print('  bcos package present')" 2>/dev/null || \
    say "  bcos NOT installed -- phase 5 will be skipped (pip install bcos)"

  say "===== preflight done; read the four checks above before committing ====="
  return 0
}

# ============================================================== PHASE 1 letterbox
stage_letterbox() {
  guard_idle || return 1
  say "===== PHASE 1  letterbox anisotropy control ====="
  say "  baseline blackbox_mobilevit_120ep egg-masked = 0.6806"
  say "  well below -> part of the shortcut was anisotropy; SEC 5.5 revises DOWN"
  say "  near 0.68  -> anisotropy not contributing; 45-69% stands, caveat discharged"
  python -u preflight_learns.py --config configs/generated/whole_blackbox_lb_120ep.yaml \
      --device cuda >> "$MASTER" 2>&1 || true; tail -4 "$MASTER"
  train_list 1 whole_blackbox_lb_120ep
  if [ -f runs/whole_blackbox_lb_120ep/best.pt ]; then
    python -u eval_accuracy_only.py \
        --config configs/generated/whole_blackbox_lb_120ep.yaml \
        --ckpt runs/whole_blackbox_lb_120ep/best.pt 2>&1 | tee -a "$MASTER"
    python -u probe_occlusion_v2.py \
        --config configs/generated/whole_blackbox_lb_120ep.yaml \
        --ckpt runs/whole_blackbox_lb_120ep/best.pt \
        --labels ../Data/Chula-ParasiteEgg-11/labels.json --device cuda \
        --emit-tsv runs/occlusion_sweep.tsv >> "$MASTER" 2>&1 || true; tail -16 "$MASTER"
    python -u probe_displaced_control.py \
        --config configs/generated/whole_blackbox_lb_120ep.yaml \
        --ckpt runs/whole_blackbox_lb_120ep/best.pt \
        --labels ../Data/Chula-ParasiteEgg-11/labels.json --device cuda \
        --emit-tsv runs/displaced_control.tsv >> "$MASTER" 2>&1 || true; tail -12 "$MASTER"
  fi
  say "===== phase 1 done ====="
}

# ============================================================= PHASE 2 multiscale
stage_multiscale() {
  guard_idle || return 1
  say "===== PHASE 2  multi-scale prototypes (-P $PAR_TRAIN) ====="
  say "  ms_coarse is the WIRING CONTROL -- must reproduce roi477_protopnet"
  say "  ms_fine tests the RF model: stride-8 prototypes should beat the ~5.5 ceiling"
  train_list "$PAR_TRAIN" "${MS_ARMS[@]}"
  probe_list "$PAR_TRAIN" "${MS_ARMS[@]}"
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import csv, glob, collections, statistics as st, os
d = collections.defaultdict(lambda: collections.defaultdict(list))
for p in glob.glob("figs/gradattr_*.tsv") + ["figs/gradattr.tsv"]:
    if not os.path.exists(p): continue
    for r in csv.DictReader(open(p), delimiter="\t"):
        try: v = float(r["mass"])
        except (ValueError, KeyError): continue
        if v == v: d[r["run"]][r["variant"]].append(v)
print(f"\n{'run':<30}{'native':>9}{'gradattr':>10}{'per_comp':>10}{'n':>5}")
for run in sorted(d):
    if "ms_" not in run and "protopnet_120ep" not in run: continue
    v = d[run]
    f = lambda k: st.mean(v[k]) if v.get(k) else float("nan")
    print(f"{run:<30}{f('native'):>9.2f}{f('gradattr'):>10.2f}"
          f"{f('per_component'):>10.2f}{len(v.get('native', [])):>5}")
print("""
  ms_coarse ~ roi477_protopnet -> wiring is right
  ms_fine   >  ~5.5            -> the receptive-field model holds; fine prototypes
                                  can resolve sub-egg structures
  ms_fine   ~= ms_coarse       -> the RF model is WRONG; withdraw MECHANISTIC SEC 3""")
PY
  say "===== phase 2 done ====="
}

# ================================================================== PHASE 3 cheap
stage_cheap() {
  guard_idle || return 1
  say "===== PHASE 3  cheap fixes ====="
  ./run_night8.sh audit >> "$MASTER" 2>&1 || true; tail -12 "$MASTER"
  ./run_night8.sh bcos  2>&1 | tail -8  | tee -a "$MASTER"
  ./run_night8.sh push  >> "$MASTER" 2>&1 || true; tail -12 "$MASTER"
  say "===== phase 3 done ====="
}

# ================================================================== PHASE 4 faith
stage_faith() {
  guard_idle || return 1
  say "===== PHASE 4  faithfulness under gradient attribution (-P $PAR_EVAL) ====="
  say "  deletion BETTER -> gradient read-out wins on both axes"
  say "  deletion WORSE  -> localisation and faithfulness genuinely trade off"
  for a in "${FAITH[@]}"; do
    [ -f "runs/$a/results.json" ] && \
      cp "runs/$a/results.json" "runs/$a/results.json.native-attr"
    rm -f "runs/$a/.eval_complete"
  done
  printf '%s\n' "${FAITH[@]}" | xargs -P "$PAR_EVAL" -I{} bash -c '
    a="{}"
    [ -f "runs/$a/best.pt" ] || exit 0
    echo "  start $a"
    PXAI_GRADATTR=1 PXAI_GRADATTR_SMOOTH=1 \
      python -u -m pxai.evaluate --config "configs/generated/$a.yaml" \
        --ckpt "runs/$a/best.pt" > "runs/$a/eval_gradattr.log" 2>&1 \
      && echo "  done  $a" || echo "  FAIL  $a"' 2>&1 | tee -a "$MASTER"
  python - <<'PY' 2>&1 | tee -a "$MASTER"
import json, os
print(f"\n  {'run':<26}{'method':<18}{'native':>9}{'gradattr':>10}{'':>8}")
for a in ("roi477_protopnet_120ep", "roi477_cbm_sup_120ep", "roi477_bcos_120ep"):
    n, g = f"runs/{a}/results.json.native-attr", f"runs/{a}/results.json"
    if not (os.path.exists(n) and os.path.exists(g)): continue
    dn, dg = json.load(open(n)), json.load(open(g))
    for m in dn.get("methods", {}):
        if not m.startswith("ours:"): continue
        x = dn["methods"][m]["faithfulness"].get("deletion", float("nan"))
        y = dg.get("methods", {}).get(m, {}).get("faithfulness", {}).get(
            "deletion", float("nan"))
        print(f"  {a:<26}{m:<18}{x:>9.4f}{y:>10.4f}{'better' if y < x else 'worse':>8}")
PY
  say "===== phase 4 done ====="
}

# ================================================================ PHASE 5 bcosnet
stage_bcosnet() {
  guard_idle || return 1
  say "===== PHASE 5  real B-cos network ====="
  if ! python -c "import bcos" 2>/dev/null; then
    say "  bcos not installed; skipping. pip install bcos"
    return 0
  fi
  say "  PREDICTION: a real B-cos net is exactly linear in x, so grad x input IS the"
  say "  native explanation and gradattr gain should be ~1.0x. The B-cos-STYLE head"
  say "  gives 2.2x. gain ~1.0 validates Part II by a correct NULL prediction."
  [ -f pxai/models/bcos_backbone.py ] || cp bcos_backbone.py pxai/models/
  # `_smoke()` IS main -- there is no --smoke flag. And no `| tail | tee`: tail closes
  # the pipe early, python takes SIGPIPE, and with `set -o pipefail` the stage dies
  # silently. That is what killed the letterbox stage at 13:42.
  python -u pxai/models/bcos_backbone.py --device cuda >> "$MASTER" 2>&1 || true
  tail -14 "$MASTER"
  say "  (registering the backbone in models/__init__.py is a manual step -- see the"
  say "   'NEXT IF THIS PASSES' block printed above; not automated)"
  say "===== phase 5 done ====="
}

case "${1:-all}" in
  preflight)  stage_preflight ;;
  letterbox)  stage_letterbox ;;
  multiscale) stage_multiscale ;;
  cheap)      stage_cheap ;;
  faith)      stage_faith ;;
  bcosnet)    stage_bcosnet ;;
  all)        stage_preflight
              stage_letterbox
              stage_multiscale
              stage_cheap
              stage_faith
              stage_bcosnet ;;
  *)          echo "usage: $0 [all|preflight|letterbox|multiscale|cheap|faith|bcosnet]"
              exit 2 ;;
esac

say "done. master log: $MASTER"
