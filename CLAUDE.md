# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This is a **benchmark scaffold**, not a validated scientific result. The model architecture, splits, training loop, and tests exist, but conclusiveness depends on running real Transition1x training across seeds 17/29/43 with leakage-safe family splits. Treat any results in `benchmarks/benchmark_results.csv` as scaffold output unless trained on real data. See `README.md` for the full conclusiveness checklist.

## Setup & Common Commands

```bash
pip install -e ".[chem,mace]"          # full install (h5py, rdkit, mace-torch)
pytest -q                              # run all tests
pytest tests/test_model_clean.py -v    # single test file
pytest tests/test_invariances.py::test_charge_conservation -v  # single test

# Generate splits (leakage-safe, by formula family)
python -m ralrc.split --h5 data/transition1x.h5 --out splits.json --seed 17
python -m ralrc.split --h5 data/transition1x.h5 --verify splits.json

# Train one ablation
python -m ralrc.train --config configs/learned_charge_coulomb.yaml --seed 17 \
    --h5 data/transition1x.h5 --splits splits.json

# Evaluate a checkpoint
python -m ralrc.eval --config configs/learned_charge_coulomb.yaml \
    --checkpoint runs/learned_charge_coulomb/seed17/best.pt \
    --h5 data/transition1x.h5 --splits splits.json \
    --out benchmarks/benchmark_results.csv --timing

# Run the full ablation grid (4 models x 3 seeds, train + eval)
bash scripts/run_all_ablations.sh
```

CLI entry points (from `pyproject.toml`): `ralrc-train`, `ralrc-eval`, `ralrc-split`, `ralrc-md`.

## Architecture

The model is `ChargeAwarePotentialClean` in `src/ralrc/model_clean.py`. Total energy:

```
E_total  = Σ_i E_i^local  +  λ_coul · E_coul
E_coul   = k_e · Σ_{i<j} q_i q_j / sqrt(r_ij² + softplus(a_{ZiZj})²)   # shielded
q_i      = q_raw_i + (Q − Σ_j q_raw_j) / N                              # exact charge conservation
F_i      = −∇_{R_i} E_total                                              # autograd, conservative
```

Key invariants the code is designed around:

- **`forward_energy()` is the pure differentiable core**; `forward()` calls it then runs `autograd.grad` once. Forces are never computed in `__init__` or via property setters. `compute_forces=False` is safe under `torch.no_grad()`.
- **No `torch.cdist`** — pairwise distances use explicit broadcasting (`R.unsqueeze(1) - R.unsqueeze(0)`) because `_cdist_backward` is missing on some CUDA builds (e.g. torch 2.11+cu128). Don't reintroduce `cdist`.
- **Charge conservation is exact** by construction (the additive correction term), not a soft loss. Tests in `tests/test_invariances.py` enforce this and translation/rotation/permutation invariance.
- `Z`, `Q`, `S` must be `torch.long` at the boundary; `_validate_inputs` enforces this.

## Compound-Key Splits (critical)

Splits use **compound keys** `"<hdf5_split>::<formula>::<rxn_id>"` — never bare `rxn_id`. Bare ids collide across formulas and across the HDF5's own train/val/test groups, which silently leaks data.

- `src/ralrc/split.py` partitions by **formula family**: 70% train / 15% val / 10% test_id_same_family / 5% test_ood_family. Held-out families go to OOD.
- `src/ralrc/train.py::_compound_key` and the same helper in `eval.py` rebuild keys from the dataset index entry `(split, formula, rxn_id, frame_idx, endpoint)`.
- `train.py` hard-fails if zero train or val frames match — this almost always means stale `splits.json` vs current HDF5. Regenerate with `ralrc.split`.
- `Transition1xDataset(h5_path, splits=None)` in `transition1x.py` builds an index over **all HDF5 splits** so every compound key is visible during filtering. Don't restrict `splits=` at the dataset level for training.

## Data Layer

`src/ralrc/transition1x.py` is the only HDF5 reader. It is **lazy** — `Transition1xDataset` builds an in-memory index of `(split, formula, rxn_id, frame_idx, endpoint)` tuples and loads array data only on `__getitem__`. Don't materialize the full dataset.

Energies/forces are stored in **Hartree** and Hartree/Å. Training converts to eV via `HA_TO_EV = 27.2114` in `train.py::_sample_to_tensors`. Keep this conversion at the boundary — model and loss are eV-native.

Required HDF5 keys per reaction group: `atomic_numbers`, `positions`, `wB97x_6-31G(d).energy`, `wB97x_6-31G(d).forces`. Groups missing any of these are skipped silently by both the iterator and the dataset index.

## Ablation Matrix

Four configs in `configs/` flip the two axes of interest:

| config                          | use_charge | use_coulomb |
|---------------------------------|------------|-------------|
| `local_mace_style.yaml`         | false      | false       |
| `charge_head_no_coulomb.yaml`   | true       | false       |
| `fixed_charge_coulomb.yaml`     | false*     | true        |
| `learned_charge_coulomb.yaml`   | true       | true        |

`use_charge=false` freezes `charge_head` parameters and `shield` (see `train.py` lines ~193–196). The same YAML files appear under `configs/baselines/` and `configs/ablations/` — the top-level copies are what `run_all_ablations.sh` uses.

## Outputs

- `runs/<model_name>/seed<N>/best.pt` — checkpoint with lowest val force MAE, plus `log.jsonl` and `summary.json`.
- `benchmarks/benchmark_results.csv` — appended one row per `(model, seed)` by `ralrc.eval`. Columns include `energy_mae`, `force_mae`, `barrier_mae`, `ts_force_mae`, `ood_degradation`, `runtime_per_atom_step`.
- `eval.py` hard-fails if `n_id_frames == 0` or `n_ood_frames == 0` rather than writing NaN — same root cause as the training case (stale splits).

## Things That Are Easy to Get Wrong

- Don't use `torch.cdist` (see above).
- Don't strip the `(Q - Σq_raw) / N` correction — it's the charge-conservation guarantee that the invariance tests check.
- Don't restrict `Transition1xDataset(splits=...)` during training; filtering happens via compound-key membership against the full index.
- Don't write bare `rxn_id` strings into `splits.json`. `train.py` detects this via the `"::" in key` check and refuses to start.
- Energies in HDF5 are Hartree; remember the `HA_TO_EV` conversion when adding new loss terms or eval metrics.
