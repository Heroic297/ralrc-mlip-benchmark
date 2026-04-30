#!/usr/bin/env bash
# Preflight check for seed17 ablation — uses real data, tiny split, 2 epochs.
#
# Purpose: verify the full train→eval pipeline runs without NaN/Inf or zero
# frames before committing to 150-epoch full training.
#
# NOT the full benchmark. Writes to:
#   runs/<model>/seed17/best.pt    (will be overwritten by full training)
#   reports/preflight_seed17_results.csv
#
# Usage:
#   bash scripts/run_preflight_seed17.sh
#
# Set H5 or SPLITS env vars to override defaults.

set -euo pipefail

SEED=17
H5="${H5:-data/transition1x.h5}"
SPLITS="${SPLITS:-splits_preflight_seed17.json}"
EPOCHS=2
OUT_CSV=reports/preflight_seed17_results.csv
MODELS=(local_mace_style charge_head_no_coulomb fixed_charge_coulomb learned_charge_coulomb)

# --- Guard rails ---
if [ ! -f "$H5" ]; then
    echo "ERROR: HDF5 not found at $H5"
    exit 1
fi

if [ ! -f "$SPLITS" ]; then
    echo "ERROR: Split file not found at $SPLITS"
    echo "  Generate it with:"
    echo "  python -m ralrc.split --h5 $H5 --out $SPLITS --seed $SEED --max-reactions 20"
    exit 1
fi

mkdir -p reports

echo "=== RALRC preflight — seed $SEED | $EPOCHS epochs | $(basename $SPLITS) ==="
echo "  H5     : $H5"
echo "  SPLITS : $SPLITS"
echo "  OUT    : $OUT_CSV"
echo ""

# Remove stale preflight CSV so each run is fresh
rm -f "$OUT_CSV"

# --- Train all four models ---
for MODEL in "${MODELS[@]}"; do
    echo "=== [train] $MODEL  seed=$SEED  epochs=$EPOCHS ==="
    python -m ralrc.train \
        --config "configs/${MODEL}.yaml" \
        --seed "$SEED" \
        --h5 "$H5" \
        --splits "$SPLITS" \
        --epochs "$EPOCHS"
    echo ""
done

# --- Eval all four models ---
for MODEL in "${MODELS[@]}"; do
    CKPT="runs/${MODEL}/seed${SEED}/best.pt"
    if [ -f "$CKPT" ]; then
        echo "=== [eval] $MODEL  seed=$SEED ==="
        python -m ralrc.eval \
            --config "configs/${MODEL}.yaml" \
            --checkpoint "$CKPT" \
            --h5 "$H5" \
            --splits "$SPLITS" \
            --out "$OUT_CSV" \
            --timing
        echo ""
    else
        echo "ERROR: checkpoint missing after training: $CKPT"
        exit 1
    fi
done

# --- Sanity check: all four rows present, no nan ---
echo "=== Preflight CSV ==="
cat "$OUT_CSV"
echo ""

ROWS=$(tail -n +2 "$OUT_CSV" | wc -l)
if [ "$ROWS" -ne 4 ]; then
    echo "ERROR: expected 4 result rows, got $ROWS"
    exit 1
fi

NAN_COUNT=$(grep -c "nan" "$OUT_CSV" || true)
if [ "$NAN_COUNT" -gt 0 ]; then
    echo "WARNING: $NAN_COUNT 'nan' values found in results"
else
    echo "OK: no nan values in results"
fi

echo ""
echo "=== Preflight PASSED ==="
echo "  Results: $OUT_CSV"
echo ""
echo "To run full seed17 training (150 epochs, full split), use:"
echo "  bash scripts/run_ablations_seed17.sh"
