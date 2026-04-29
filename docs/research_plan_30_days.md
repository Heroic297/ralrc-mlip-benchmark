# RALRC Benchmark: 30-Day Research Plan

> Status: BENCHMARK SCAFFOLD — no training completed as of 2026-04-29.
> This is a falsifiable benchmark study, NOT a SOTA claim.

---

## Research Question

Does conditioning a MACE-style message-passing potential on total charge Q —
and adding a learned shielded Coulomb correction — measurably reduce:

1. **Transition-state (TS) force MAE** on held-out reactions?
2. **Barrier height MAE** on held-out reactions?
3. **OOD degradation** (MAE_OOD / MAE_ID) compared with a local-only baseline?

at equal training budget and on the same Transition1x reaction paths?

The study accepts a negative result: if the improvements are below threshold
or not reproducible, the conclusion is "learned charge + Coulomb correction
does not help for this task/dataset at this model size."

---

## Hypothesis

**H1 (primary):** The learned-charge + Coulomb model (`learned_charge_coulomb`)
achieves ≥15 % lower TS force MAE and ≥20 % lower barrier MAE vs the local
baseline (`local_mace_style`) when trained on the full Transition1x dataset.

**H2 (secondary):** The OOD degradation ratio (MAE_OOD / MAE_ID) is smaller
for `learned_charge_coulomb` than for `local_mace_style`, indicating better
generalization to held-out reaction families.

**H3 (ablation):** The gain is attributable to the Coulomb correction
(`fixed_charge_coulomb` and `learned_charge_coulomb`) rather than the charge
head alone (`charge_head_no_coulomb`).

---

## Ablation Matrix

| Config | use_charge | use_coulomb | Purpose |
|--------|-----------|-------------|---------|
| `local_mace_style` | false | false | Baseline: local forces only |
| `charge_head_no_coulomb` | true | false | Charge conditioning, no long-range |
| `fixed_charge_coulomb` | false | true | Coulomb with frozen charge head |
| `learned_charge_coulomb` | true | true | Full model (primary hypothesis) |

All four configs use identical hyperparameters (hidden=64, epochs=150,
lr=1e-3, w_E=1, w_F=100) for a controlled ablation.

---

## Metrics

All metrics reported on Transition1x wB97x/6-31G(d) labels in eV / eV/Å:

| Metric | Units | Description |
|--------|-------|-------------|
| `energy_mae_id` | eV/molecule | Mean absolute energy error, ID test set |
| `force_mae_id` | eV/Å | Mean absolute force component error, ID |
| `ts_force_mae_id` | eV/Å | Force MAE restricted to TS-neighbourhood frames (±10% of IRC) |
| `barrier_mae_id` | eV | |E_max − E_min| MAE per reaction, ID |
| `energy_mae_ood` | eV/molecule | Same metrics on held-out OOD families |
| `force_mae_ood` | eV/Å | |
| `ts_force_mae_ood` | eV/Å | |
| `barrier_mae_ood` | eV | |
| `ood_degradation_force` | — | force_mae_ood / force_mae_id |
| `ood_degradation_barrier` | — | barrier_mae_ood / barrier_mae_id |
| `runtime_ratio_vs_local` | — | wall-clock eval time vs local_mace_style |

---

## Success Criteria

Declare a **positive result** only if ALL of the following hold across ≥2 of
3 seeds:

1. `ts_force_mae_id(learned_charge_coulomb)` < 0.85 × `ts_force_mae_id(local_mace_style)`
2. `barrier_mae_id(learned_charge_coulomb)` < 0.80 × `barrier_mae_id(local_mace_style)`
3. `ood_degradation_force(learned_charge_coulomb)` < `ood_degradation_force(local_mace_style)`
4. The improvement is larger than the gap between `charge_head_no_coulomb` and
   `local_mace_style` (i.e. Coulomb adds signal beyond charge conditioning alone)
5. `runtime_ratio_vs_local` ≤ 2.0

---

## Negative-Result Criteria

Declare a **negative result** (publishable as benchmark) if, after full training
with real data on all seeds:

- The improvements in TS force MAE and barrier MAE are < 10 % vs local baseline
- OR results are not reproducible across all 3 seeds (std > 30 % of mean)
- OR `learned_charge_coulomb` performs worse than `local_mace_style` on OOD

A negative result from a leakage-safe, well-controlled benchmark is still a
valid scientific contribution.

---

## Hard Failure Criteria (stop and investigate)

Stop the experiment and do not report results if any of the following occur:

- Any invariance test (`pytest tests/`) fails after training
- `energy_mae_id` > 1000 eV (unit conversion error or untrained model)
- `ood_degradation_force` > 10× (data leakage or split bug)
- Barrier MAE > 100 eV (unit conversion error)
- Duplicate compound keys in splits.json (`verify_leakage` PASS = False)
- Train loss diverges (NaN) in > 1 of 12 runs

---

## Exact Reproduction Commands

### Validate the pipeline (seed 17 only, ~4–8 h on GPU):

```bash
pip install -e ".[chem]"

# Generate leakage-safe splits
python -m ralrc.split --h5 data/transition1x.h5 --out splits.json --seed 17

# Verify splits
python -m ralrc.split --h5 data/transition1x.h5 --verify splits.json

# Run all 4 ablations at seed 17
bash scripts/run_ablations_seed17.sh

# Check results
cat benchmarks/benchmark_results.csv
```

### Full benchmark (all seeds, ~12–24 h on GPU):

```bash
bash scripts/run_all_ablations.sh
```

### Run tests (no GPU required):

```bash
pytest -q
```

Expected output (without training data): 33 tests pass + N additional tests
from test_coulomb_gradient.py and test_split_safety.py.

---

## 30-Day Schedule

| Days | Task |
|------|------|
| 1–2 | Run `pytest -q`; fix any failing tests |
| 3–4 | Verify HDF5 access; run `python -m ralrc.split` on full dataset |
| 5–7 | Smoke-train local_mace_style for 10 epochs; check loss is decreasing |
| 8–14 | Full seed-17 run (4 ablations × 150 epochs each) |
| 15–16 | Evaluate seed-17 checkpoints; fill in row 1–4 of results table |
| 17–20 | Diagnose any hard failures; re-run if needed |
| 21–28 | Full 3-seed run (seeds 17, 29, 43) |
| 29–30 | Compute statistics; write conclusion in reports/final_report.md |

---

## Data Notes

- Dataset: Transition1x (Schreiner et al. 2022), wB97x/6-31G(d), ~10k reactions
- Download: https://zenodo.org/record/5795407
- Expected location: `data/transition1x.h5`
- Energies stored in Hartree; training converts to eV via `HA_TO_EV = 27.2114`
- Splits: 70 % train / 15 % val / 10 % test_id / 5 % test_ood, by formula family

---

## Limitations and Caveats

- The model uses a simplified O(N²) message-passing scheme, not a proper
  MACE equivariant architecture with cutoffs. This limits scalability and
  may underperform relative to a tuned MACE baseline.
- "local_mace_style" is NOT the published MACE model — it is a simplified
  distance-weighted sum-of-atoms baseline with the same training budget.
- Results should not be compared directly to published MACE, ANI, or AIMNet2
  benchmarks, which use different architectures, datasets, and training budgets.
- A positive result here would motivate a follow-up study with a full MACE
  backbone; a negative result constrains the benefit of the charge mechanism.
