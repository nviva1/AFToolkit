#!/usr/bin/env bash
# Download the example antibody-RBD crystal complexes (deposited with the
# RBD-AIM paper) into inputs/, for the score-recomputation demo.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)/inputs"
mkdir -p "$DIR"
for id in 8WFH 8WFM 9JEB; do
  if [ ! -f "$DIR/$id.pdb" ]; then
    echo "fetching $id ..."
    curl -L --fail --silent -o "$DIR/$id.pdb" "https://files.rcsb.org/download/$id.pdb"
  fi
done
echo "structures in $DIR:"
ls -1 "$DIR"
