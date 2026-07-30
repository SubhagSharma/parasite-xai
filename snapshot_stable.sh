#!/bin/bash
# snapshot_stable.sh — copy ALL of runs/ and configs/ into stable/<name>/
# so the complete current state stays accessible no matter what later jobs do.
#
#   ./snapshot_stable.sh                 # timestamped name
#   ./snapshot_stable.sh pre_night_run   # or your own name
#
# Everything is a plain copy, so browse or restore with normal cp:
#   cp stable/<name>/runs/<run>/results.json runs/<run>/

set -u
cd "$(dirname "$0")"

STAMP=${1:-$(date +%Y%m%d_%H%M)}
DEST="stable/$STAMP"

if [ -d "$DEST" ]; then
  echo "ERROR: $DEST already exists - pick another name."
  exit 1
fi

echo "copying runs/ and configs/ -> $DEST"
mkdir -p "$DEST"
cp -rp runs    "$DEST/runs"
cp -rp configs "$DEST/configs"

echo ""
du -sh "$DEST"
echo "SNAPSHOT COMPLETE: $DEST"