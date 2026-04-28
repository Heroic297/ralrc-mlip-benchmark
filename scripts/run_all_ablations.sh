#!/usr/bin/env bash
# Run all four ablation models across seeds 17, 29, 43.
# Usage: bash scripts/run_all_ablations.sh
# Prerequisites:
#   1. Transition1x.h5 at data/transition1x.h5
#   2. splits.json generated (see Step 0 below)
#   3. pip install -e .

set -euo pipefail

H5=data/transition1x.h5
SPLITS=splits.json

# -----------------------------------------------------------------------
# Step 0: Generate leakage-safe splits (idempotent)
# -----------------------------------------------------------------------
if [ ! -f "$SPLITS" ]; then
    echo "=== Generating splits ==="
    python -m ralrc.split --h5 "$H5" --out "$SPLITS" --seed 17
else
    echo "=== Verifying existing splits ==="
    python -m ralrc.split --h5 "$H5" --verify "$SPLITS"
fi

# -----------------------------------------------------------------------
# Step 1: Train all four models x 3 seeds
# -----------------------------------------------------------------------
MODELS=(local_mace_style charge_head_no_coulomb fixed_charge_coulomb learned_charge_coulomb)
SEEDS=(17 29 43)

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo
        echo "=== Training $MODEL seed=$SEED ==="
        python -m ralrc.train \
            --config "configs/${MODEL}.yaml" \
            --seed "$SEED" \
            --h5 "$H5" \
            --splits "$SPLITS"
    done
done

# -----------------------------------------------------------------------
# Step 2: Evaluate best checkpoints for each model (seed 17 primary)
# -----------------------------------------------------------------------
mkdir -p benchmarks

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT="runs/${MODEL}/seed${SEED}/best.pt"
        if [ -f "$CKPT" ]; then
            echo
            echo "=== Evaluating $MODEL seed=$SEED ==="
            python -m ralrc.eval \
                --config "configs/${MODEL}.yaml" \
                --checkpoint "$CKPT" \
                --h5 "$H5" \
                --splits "$SPLITS" \
                --out benchmarks/benchmark_results.csv \
                --timing
        else
            echo "WARNING: checkpoint not found: $CKPT"
        fi
    done
done

echo
echo "=== All done. Results in benchmarks/benchmark_results.csv ==="
cat benchmarks/benchmark_results.csv
