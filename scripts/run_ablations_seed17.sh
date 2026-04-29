#!/usr/bin/env bash
# Run all four ablation models for seed 17 only.
#
# This is the minimum necessary to fill in one row per ablation in the
# seed17 results table.  Run this before run_all_ablations.sh to validate
# the pipeline on a single seed.
#
# Prerequisites:
#   1. pip install -e ".[chem]"
#   2. Transition1x.h5 at data/transition1x.h5
#      Download: https://zenodo.org/record/5795407  (file: Transition1x.h5)
#
# Outputs:
#   runs/<model>/seed17/best.pt          checkpoint with lowest val force MAE
#   runs/<model>/seed17/log.jsonl        per-epoch training metrics
#   runs/<model>/seed17/summary.json     best epoch / MAE summary
#   benchmarks/benchmark_results.csv     eval rows (appended)
#
# Usage:
#   bash scripts/run_ablations_seed17.sh

set -euo pipefail

SEED=17
H5=data/transition1x.h5
SPLITS=splits.json
MODELS=(local_mace_style charge_head_no_coulomb fixed_charge_coulomb learned_charge_coulomb)

# ------------------------------------------------------------------
# Fail fast: data must exist before we do anything else
# ------------------------------------------------------------------
if [ ! -f "$H5" ]; then
    echo "ERROR: Transition1x data not found at $H5"
    echo ""
    echo "  Download the dataset from:"
    echo "    https://zenodo.org/record/5795407"
    echo "  Then place the .h5 file at $H5  (or set H5= to override)"
    exit 1
fi

echo "=== RALRC benchmark — seed $SEED ==="
echo "  H5   : $H5"
echo "  SPLITS: $SPLITS"

# ------------------------------------------------------------------
# Step 0: Generate or verify leakage-safe family splits
# ------------------------------------------------------------------
if [ ! -f "$SPLITS" ]; then
    echo ""
    echo "=== Step 0: Generating family splits (seed $SEED) ==="
    python -m ralrc.split --h5 "$H5" --out "$SPLITS" --seed "$SEED"
else
    echo ""
    echo "=== Step 0: Verifying existing splits ==="
    python -m ralrc.split --h5 "$H5" --verify "$SPLITS"
fi

# ------------------------------------------------------------------
# Step 1: Train all four models at seed 17
# ------------------------------------------------------------------
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=== Step 1 [train] $MODEL  seed=$SEED ==="
    python -m ralrc.train \
        --config "configs/${MODEL}.yaml" \
        --seed "$SEED" \
        --h5 "$H5" \
        --splits "$SPLITS"
done

# ------------------------------------------------------------------
# Step 2: Evaluate best checkpoints
# ------------------------------------------------------------------
mkdir -p benchmarks reports

for MODEL in "${MODELS[@]}"; do
    CKPT="runs/${MODEL}/seed${SEED}/best.pt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=== Step 2 [eval] $MODEL  seed=$SEED ==="
        python -m ralrc.eval \
            --config "configs/${MODEL}.yaml" \
            --checkpoint "$CKPT" \
            --h5 "$H5" \
            --splits "$SPLITS" \
            --out benchmarks/benchmark_results.csv \
            --timing
    else
        echo "WARNING: checkpoint not found: $CKPT  (training may have failed)"
    fi
done

echo ""
echo "=== Seed $SEED complete. ==="
echo "  Results appended to: benchmarks/benchmark_results.csv"
echo "  Checkpoints in:      runs/*/seed${SEED}/"
