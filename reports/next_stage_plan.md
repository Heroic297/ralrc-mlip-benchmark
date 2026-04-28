# RALRC MLIP Benchmark -- Next Stage Plan

## Current Status

**Tests OK but NO REAL TRAINING = INCONCLUSIVE**

The benchmark harness (`test_model_clean.py`) passes all symmetry and
consistency checks. This is a necessary but not sufficient condition for
scientific validity.

## Why Passing Tests Is Not a Scientific Result

The symmetry tests verify:
- charge conservation algebra
- translation/rotation/permutation invariance
- force finite-difference consistency
- force equivariance
- shielded Coulomb finiteness

They do NOT verify:
- that the model learns a realistic PES
- that it generalises to unseen reaction families
- that it outperforms any baseline

## Real Success Criteria

| Criterion | Threshold | Status |
|---|---|---|
| Real training data (Transition1x or SPICE) | loaded | NOT MET |
| Reaction-family OOD split | implemented | NOT MET |
| Barrier MAE vs tuned local MACE | >=20-30% improvement | NOT MET |
| TS-force MAE | >=15% improvement | NOT MET |
| OOD degradation | decreases vs baseline | NOT MET |
| Charged/ion MAE | improves without neutral regression | NOT MET |
| NVE/NVT MD stability | energy conserved, stable | NOT MET |
| Long-range Coulomb | E ~ 1/r at large separation | NOT MET |
| Runtime | within 2x local MACE | NOT MET |
| Reproducibility | seeds 17, 29, 43 | NOT MET |
| Ablation | Coulomb removal shows degradation | NOT MET |

## Next Steps

1. Acquire Transition1x or SPICE dataset.
2. Implement ASE-compatible data loader in `data.py`.
3. Implement reaction-family train/val/test split in `split.py`.
4. Train tuned local MACE baseline (Coulomb off, charge head off).
5. Train full ChargeAwarePotentialClean with same data.
6. Evaluate on barrier MAE, TS-force MAE, OOD sets.
7. Run NVE/NVT MD stability checks.
8. Run ablations: no Coulomb, no charge head, no Q conditioning.
9. Reproduce across seeds 17, 29, 43.
10. Report results in `reports/final_report.md`.

## Architecture Notes

`model_clean.py` uses a simplified O(N^2) message-passing scheme
sufficient for benchmarking correctness. For production:
- Replace `torch.cdist` with a proper neighbor list (e.g. `ase.neighborlist`)
- Replace the distance-weighted message-pass with MACE-style equivariant
  message-passing using `e3nn`
- Add cutoff envelope functions for smoothness

## Files

| File | Purpose |
|---|---|
| `src/ralrc/model_clean.py` | Clean model with stable API |
| `tests/test_model_clean.py` | Symmetry and schema tests |
| `scripts/run_clean_tests.ps1` | Install deps + run tests |
| `scripts/check_conclusiveness.ps1` | Print conclusiveness verdict |
| `reports/next_stage_plan.md` | This file |