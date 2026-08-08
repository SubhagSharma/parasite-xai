#!/usr/bin/env bash
# separate file so the wrapper's command line does not contain "pxai.train" --
# guard_idle greps `pgrep -f "pxai.(train|evaluate)"`, which would otherwise match
# the wrapper against itself and stall every stage for 5 minutes before aborting
cd "$(dirname "$0")"
mkdir -p runs/roi477_bcosnet_120ep
python -u -m pxai.train --config configs/generated/roi477_bcosnet_120ep.yaml \
  > runs/roi477_bcosnet_120ep/train.log 2>&1
python -u probe_gradattr.py --device cuda --runs roi477_bcosnet_120ep
echo "########## COMPLETE $(date) ##########"
