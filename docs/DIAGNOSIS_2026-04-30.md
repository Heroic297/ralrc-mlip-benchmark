# Diagnosis — local_mace_style 30-epoch oscillation (2026-04-30)

Run was killed at epoch 10. Train loss decreased monotonically (225k → 35k);
val F_MAE oscillated 6.6–14.3 eV/Å with no convergence; val E_MAE stuck near
10⁴ eV. Symptom is a textbook missing per-element reference energy combined
with an energy-dominated composite loss.

## Hypothesis 1 — Per-element E_ref not subtracted from targets

**CONFIRMED.**

`AtomRefEnergy` is initialized to zeros and learned via SGD only — there is
no closed-form lstsq pre-fit:

- `src/ralrc/train.py:69-76` — `AtomRefEnergy.__init__` calls
  `nn.init.zeros_(self.ref.weight)`. No pre-fit hook.
- `src/ralrc/train.py:226-227` — its parameters are added to the same Adam
  optimizer with the same `lr=1e-3` cosine schedule as the model. No
  separate, faster schedule.
- `src/ralrc/train.py:283` (training) and `src/ralrc/train.py:322`
  (validation) — target is `E_ref_corr = E_ref - atom_ref(Z)`. With
  `atom_ref` initially zero, training sees raw DFT total energies (~10⁴ eV
  per molecule on Transition1x) and must regress against them via SGD.
- `src/ralrc/transition1x.py:51-72` — dataset loads raw energies in Hartree.
  No subtraction at the data layer.

This is exactly the trajectory observed: epoch-1 E_MAE=213,854 eV → epoch-10
E_MAE=39,026 eV is a slow linear regression of the per-element bias by
gradient descent, two orders of magnitude away from where it would land if
pre-fit by `numpy.linalg.lstsq` at construction time.

## Hypothesis 2 — Loss reduction & per-atom normalization

**PARTIALLY CONFIRMED.**

`src/ralrc/train.py:90-94`:
```python
def energy_force_loss(E_pred, F_pred, E_ref, F_ref, w_E=1.0, w_F=100.0):
    return w_E * (E_pred - E_ref).abs() + w_F * (F_pred - F_ref).abs().mean()
```

- Force term: `.abs().mean()` over (N, 3) — correct per-component reduction.
- Energy term: `.abs()` of a scalar — correct per-molecule reduction, but
  **not per-atom**. Larger molecules carry larger energy errors.
- With raw DFT energies of ~10⁴ eV and uncorrected refs, the per-molecule
  energy term dominates the loss by the factor reported by the user (~41×).
  Once H1 is fixed, residual energies become small (~eV scale) and `w_E=1,
  w_F=100` is sane. Per-atom normalization is a follow-up optimization, not
  the root cause.

Concrete values:
- `w_E = 1.0`, `w_F = 100.0` (configs/local_mace_style.yaml:7-8).
- Energy loss reduction: implicit scalar (single-molecule).
- Force loss reduction: `.mean()` over (N, 3).
- E_ref pre-fit: **NO** — SGD-learned from zero.
- Target computed at `train.py:283` (train) and `train.py:322` (val).

## Hypothesis 3 — Dataloader throughput

**REFUTED as cause.** Loss is decreasing; throughput isn't the bug. Prior
session deliberately set `--num-workers 0` for Windows IPC reasons. Not
investigating further per task instructions.

## Hypothesis 4 — Python 3.14 + torch 2.11 sanity

**Verified, no anomalies.**

```
torch 2.11.0+cu128  cuda 12.8  available=True  device=NVIDIA GeForce RTX 5090 Laptop GPU
```

## Hypothesis 5 (added on review) — Unit convention

**CONFIRMED BUG.** CLAUDE.md and `train.py` claimed Transition1x stores
energies in Hartree and applied `HA_TO_EV = 27.2114` in `_sample_to_tensors`.
The HDF5 stores energies in **eV** already (Schreiner et al. 2022,
*Scientific Data*); no `units` attribute is set. Empirical check:
C₂H₂N₂O₂ raw E = -9176.34 vs sum of per-atom ωB97X/6-31G(d) = -9132 eV
(ratio 1.005, vs Hartree expectation ratio 27.34). The spurious ×27.2114
inflated energies and forces by ~27× without breaking MAE comparisons
(uniform multiplier), but produced misleading reported magnitudes.
`HA_TO_EV` is now `1.0` and refs are fitted directly in eV; literature
single-atom ωB97X/6-31G(d) energies match the lstsq output to ~1%.

## Fix

1. `scripts/fit_atomic_refs.py` solves `E_total ≈ X @ e_ref` via
   `numpy.linalg.lstsq` on TRAIN-split frames; writes `runs/e_ref.json`
   keyed by atomic number (eV).
2. `Transition1xDataset` accepts `atom_refs` and subtracts
   `Σ_Z n_Z · e_ref_Z` from the energy target in `__getitem__` (in Hartree).
   Train and eval pass the same refs → symmetric.
3. `AtomRefEnergy` is removed from the training loss path; its job is now
   done at data-load time with closed-form values.
4. `--smoke-test` flag in `ralrc.train`: 2k train / 500 val / 1 epoch / bs=64,
   asserts E_MAE < 5.0 eV/atom and F_MAE < 3.0 eV/Å.
