#!/usr/bin/env bash
# Full benchmark: 4 models x 5 folds on one GPU, then the summary table.
# Each (model, fold) is skipped if runs/<model>/fold<k>/metrics.json already exists, so the
# script can be stopped and restarted. Run from the repo root.
set -euo pipefail
mkdir -p runs results

MODELS=${MODELS:-"resnet50 vit_b16 cnn_thesis vit_thesis"}

for model in $MODELS; do
    for fold in 0 1 2 3 4; do
        if [[ -f "runs/$model/fold$fold/metrics.json" ]]; then
            echo "skip $model fold$fold"
            continue
        fi
        uv run python -m thyroid.train --model "$model" --fold "$fold" 2>&1 | tee -a "runs/train.log"
    done
done

uv run python -m thyroid.summarize | tee results/summary.md
