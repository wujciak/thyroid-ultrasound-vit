#!/usr/bin/env bash
# Downloads the thyroid ultrasound dataset (Figshare, CC BY 4.0) into data/raw/figshare.
# Run from the repo root, then: uv run python -m thyroid.data.prepare
set -euo pipefail

OUT=data/raw/figshare
mkdir -p "$OUT"

fetch() {
    local name=$1 id=$2 md5=$3
    if [[ ! -f "$OUT/$name" ]] || ! echo "$md5  $OUT/$name" | md5sum -c --quiet; then
        curl -fL -o "$OUT/$name" "https://ndownloader.figshare.com/files/$id"
        echo "$md5  $OUT/$name" | md5sum -c
    fi
}

# https://doi.org/10.6084/m9.figshare.26067475
fetch batch1_image.zip 47150848 59a9f5dab483339c78ece7bc3d7f6d9f
fetch batch2_image.zip 47150851 6cc805106b8b82851201cdf8f9c0281d
