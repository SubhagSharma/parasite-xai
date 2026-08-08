#!/usr/bin/env bash
# run_anatomy_scale.sh — does anatomical consistency scale with how many pixels the
# feature occupies?
#
#   nohup ./run_anatomy_scale.sh > logs/anatomy_scale.log 2>&1 &
#
# PHASE 1  train concept-part heads on crops, roi679, whole   ~3.0h  (-P 3)
# PHASE 2  train it on a frozen DINOv2 backbone               ~1.0h
# PHASE 3  anatomy tests on all five arms                     ~0.5h
#                                                      TOTAL  ~4.5h
#
# THE QUESTION
# probe_anatomy found that concept slots learn CONSISTENT, CAUSAL, PER-SPECIES evidence
# regions but not the named anatomy: `operculum` had within-species centroid spread
# 0.235 and between-species 0.453, and the figures show it firing on the egg's top edge
# for Fasciolopsis and on adjacent background for Paragonimus. Only 2 of 12 multi-species
# concepts were anatomically consistent.
#
# The obvious candidate cause is resolution. Feature size against the stride-32 grid:
#
#   dataset   egg % frame   egg diameter   polar plug (~22% of egg)   vs a 32px cell
#   whole         2%           32 px             7 px                   0.2 cells
#   roi679      5.8%           54 px            12 px                   0.4 cells
#   roi477       12%           77 px            17 px                   0.5 cells
#   crops        55%          166 px            37 px                   1.2 cells
#
# On crops a polar plug crosses one feature cell for the FIRST TIME. So the prediction
# is not merely "crops will be better" but a DOSE-RESPONSE: anatomical consistency
# should improve monotonically with egg fraction, with a threshold near one cell.
# Four points on that curve is a much stronger result than any pairwise comparison.
#
# WHY THE CROP CAVEAT DOES NOT APPLY HERE
# `conc_pos` saturates on crops because the annotation box covers 55% of the frame, so
# crop rows are excluded from every localisation table in the report. But all three
# anatomy tests are BOX-RELATIVE -- centroid consistency, between/within ratio, and the
# polar/central/peripheral priors are computed in coordinates where (0,0) is the box's
# top-left corner. None of them saturate; if anything they are better resolved on crops.
#
# WHY DINOv2 AND NOT CLIP
# CLIP optimises a global image-text alignment; its patch features are weaker spatially.
# DINOv2 is trained with dense objectives and is the stronger choice for localisation.
# P2 already showed DINOv2 carries 3.6x the within-class rank of the supervised backbone
# but that PROTOTYPE rank did not follow -- concepts are SUPERVISED where prototypes are
# DISCOVERED, so the mechanism differs and the outcome may too. CLIP's distinct value is
# its text encoder, which would let concepts be specified by prompt (Label-free CBM,
# Oikarinen et al.) -- that tests concept SPECIFICATION, not localisation, and is a
# separate experiment.
#
# PREDICTIONS, RECORDED BEFORE THE RUN
#   between/within centroid ratio falls monotonically: whole > roi679 > roi477 > crops
#   the polar/central/peripheral priors separate on crops and not on whole images
#   if crops show NO improvement, resolution is not the cause and the failure is about
#     what the concept supervision can express -- also a result, and a cleaner one

set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs figs
MASTER="logs/anatomy_scale_$(date +%Y%m%d_%H%M).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

ARMS=(crop_parts_120ep roi679_parts_120ep whole_parts_120ep)
ALL=(roi477_parts_120ep "${ARMS[@]}" dino_parts_120ep)

wait_idle() {
  for _ in $(seq 1 90); do
    [ "$(pgrep -fc 'pxai.train|pxai.evaluate' || true)" -eq 0 ] && return 0
    sleep 60
  done
  say "WARN: still busy; continuing"
}

train_one() {
  local a="$1"
  [ -f "configs/generated/$a.yaml" ] || { say "  skip $a (no config)"; return 0; }
  [ -f "runs/$a/.train_complete" ] && { say "  skip $a (done)"; return 0; }
  mkdir -p "runs/$a"
  say "  train $a"
  python -u -m pxai.train --config "configs/generated/$a.yaml" \
      > "runs/$a/train.log" 2>&1 \
    && touch "runs/$a/.train_complete" \
    && say "  done  $a  $(grep -o 'test_acc=[0-9.]*' runs/$a/train.log | tail -1)" \
    || say "  FAIL  $a"
}

echo "########## ANATOMY SCALE STARTED $(date) ##########" | tee -a "$MASTER"

# ================================================================= PHASE 0 configs
say "===== configs ====="
python - >> "$MASTER" 2>&1 <<'PYC'
import yaml, copy, os

base = yaml.safe_load(open("configs/generated/roi477_parts_120ep.yaml"))
roots = {
    "crop_parts_120ep":   "../Data/chula_crops",
    "roi679_parts_120ep": "../Data/chula_roi2_w679",
    "whole_parts_120ep":  "../Data/Chula-ParasiteEgg-11/data",
}
for name, root in roots.items():
    if not os.path.exists(root):
        print(f"  SKIP {name}: {root} missing"); continue
    c = copy.deepcopy(base)
    c["data"]["root"] = root
    c["output_dir"] = f"./runs/{name}"
    yaml.safe_dump(c, open(f"configs/generated/{name}.yaml", "w"), sort_keys=False)
    print(f"  {name:<24} root {root}")

# frozen DINOv2: dense self-supervised features, 3.6x the within-class rank of the
# supervised backbone (P1). Concepts are SUPERVISED, unlike prototypes, so the P2 null
# does not necessarily transfer.
if os.path.exists("pxai/models/dino_backbone.py"):
    c = copy.deepcopy(base)
    c["backbone"]["name"] = "dinov2_vits14"
    c["backbone"]["pretrained"] = True
    c["output_dir"] = "./runs/dino_parts_120ep"
    yaml.safe_dump(c, open("configs/generated/dino_parts_120ep.yaml", "w"),
                   sort_keys=False)
    print("  dino_parts_120ep         backbone dinov2_vits14 (frozen)")
else:
    print("  SKIP dino_parts_120ep: pxai/models/dino_backbone.py missing")
PYC
tail -6 "$MASTER"

# ================================================================= PHASE 1 pixels
say "===== PHASE 1  three pixel scales (-P 3) ====="
say "  crops 55% of frame, roi679 5.8%, whole 2%. roi477 (12%) is already trained."
wait_idle
printf '%s\n' "${ARMS[@]}" | xargs -P 3 -I{} bash -c '
  a="{}"
  [ -f "configs/generated/$a.yaml" ] || { echo "  skip $a"; exit 0; }
  [ -f "runs/$a/.train_complete" ] && { echo "  skip $a (done)"; exit 0; }
  mkdir -p "runs/$a"; echo "  start $a"
  python -u -m pxai.train --config "configs/generated/$a.yaml" \
    > "runs/$a/train.log" 2>&1 && touch "runs/$a/.train_complete" \
    && echo "  done  $a" || echo "  FAIL  $a"' 2>&1 | tee -a "$MASTER"

# ================================================================== PHASE 2 dino
say "===== PHASE 2  frozen DINOv2 backbone ====="
wait_idle
train_one dino_parts_120ep

# ================================================================ PHASE 3 anatomy
say "===== PHASE 3  anatomy tests on every arm ====="
for a in "${ALL[@]}"; do
  [ -f "runs/$a/best.pt" ] || { say "  skip $a (no checkpoint)"; continue; }
  say "-- $a --"
  python -u probe_anatomy.py --run "$a" --device cuda >> "$MASTER" 2>&1 || true
  grep -A 3 "concepts land in a consistent place" "$MASTER" | tail -4
  grep -A 6 "group  *|y-0.5|" "$MASTER" | tail -6
done

say "===== the dose-response curve ====="
python - >> "$MASTER" 2>&1 <<'PYD'
import re, os
# egg as a fraction of frame, measured from the annotation boxes in each dataset
FRAC = {"whole_parts_120ep": 0.020, "roi679_parts_120ep": 0.058,
        "roi477_parts_120ep": 0.118, "crop_parts_120ep": 0.538,
        "dino_parts_120ep": 0.118}
log = sorted([f for f in os.listdir("logs") if f.startswith("anatomy_scale_")])[-1]
txt = open(os.path.join("logs", log)).read()
print(f"\n  {'arm':<24}{'egg %':>8}{'plug px':>9}{'anatomical':>13}{'polar-central':>15}")
for arm in ("whole_parts_120ep", "roi679_parts_120ep", "roi477_parts_120ep",
            "crop_parts_120ep", "dino_parts_120ep"):
    seg = txt.split(f"-- {arm} --")
    if len(seg) < 2:
        continue
    s = seg[1]
    m = re.search(r"(\d+)/(\d+) concepts land in a consistent place", s)
    p = re.search(r"polar minus central polarity: ([+-][\d.]+)", s)
    f = FRAC.get(arm, float("nan"))
    plug = (f ** 0.5) * 224 * 0.22          # plug is roughly 22% of the egg's diameter
    print(f"  {arm:<24}{f*100:>7.1f}%{plug:>8.0f}px"
          f"{(m.group(0).split()[0] if m else '--'):>13}"
          f"{(p.group(1) if p else '--'):>15}")
print("""
  A monotone rise in the anatomical fraction with egg size confirms that the failure is
  RESOLUTION: a 7px polar plug on whole images cannot be found by a head reading a 32px
  feature grid, and a 37px one on crops can.

  Flat across all four -> resolution is NOT the cause. The failure would then be about
  what class-level concept supervision can express: a slot only has to produce evidence
  that DISCRIMINATES, and adjacent background discriminates just as well as the named
  structure. That is a cleaner finding and it points at per-image part annotation as the
  only fix.""")
PYD
tail -20 "$MASTER"

echo "########## COMPLETE $(date) ##########" | tee -a "$MASTER"
say "master log: $MASTER"
