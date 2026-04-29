#!/usr/bin/env bash
# Run all four ablation models across seeds 17, 29, 43.
#
# Prerequisites:
#   1. pip install -e ".[chem]"
#   2. Transition1x.h5 at data/transition1x.h5
#      Download: https://zenodo.org/record/5795407  (file: Transition1x.h5)
#
# Outputs:
#   runs/<model>/seed<N>/best.pt         checkpoints
#   runs/<model>/seed<N>/log.jsonl       per-epoch training metrics
#   runs/<model>/seed<N>/summary.json    best epoch / MAE
#   benchmarks/benchmark_results.csv     eval rows (appended)
#   reports/run_all_ablations.log        captured stdout+stderr from this run
#
# Usage:
#   bash scripts/run_all_ablations.sh
#
# To run seed 17 only first (recommended to validate pipeline):
#   bash scripts/run_ablations_seed17.sh

set -euo pipefail

H5=data/transition1x.h5
SPLITS=splits.json
MODELS=(local_mace_style charge_head_no_coulomb fixed_charge_coulomb learned_charge_coulomb)
SEEDS=(17 29 43)

# ------------------------------------------------------------------
# Fail fast: data must exist before anything else
# ------------------------------------------------------------------
if [ ! -f "$H5" ]; then
    echo "ERROR: Transition1x data not found at $H5"
    echo ""
    echo "  Download from: https://zenodo.org/record/5795407  (Transition1x.h5)"
    echo "  Then place at: $H5  (or edit H5= above)"
    exit 1
fi

mkdir -p benchmarks reports

echo "=== RALRC benchmark — all seeds (17, 29, 43) ==="
echo "  H5     : $H5"
echo "  SPLITS : $SPLITS"
echo "  Log    : reports/run_all_ablations.log"

# ------------------------------------------------------------------
# Step 0: Generate or verify leakage-safe splits (uses seed 17)
# ------------------------------------------------------------------
if [ ! -f "$SPLITS" ]; then
    echo ""
    echo "=== Step 0: Generating family splits (seed 17) ==="
    python -m ralrc.split --h5 "$H5" --out "$SPLITS" --seed 17
else
    echo ""
    echo "=== Step 0: Verifying existing splits ==="
    python -m ralrc.split --h5 "$H5" --verify "$SPLITS"
fi

# ------------------------------------------------------------------
# Step 1: Train all four models × 3 seeds
# ------------------------------------------------------------------
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "=== Step 1 [train] $MODEL  seed=$SEED ==="
        python -m ralrc.train \
            --config "configs/${MODEL}.yaml" \
            --seed "$SEED" \
            --h5 "$H5" \
            --splits "$SPLITS"
    done
done

# ------------------------------------------------------------------
# Step 2: Evaluate best checkpoints for all models × seeds
# ------------------------------------------------------------------
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
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
done

echo ""
echo "=== All done. ==="
echo "  Results in: benchmarks/benchmark_results.csv"
echo ""
cat benchmarks/benchmark_results.csv
