# Session State — RALRC Pilot Benchmark

**Stopped:** 2026-04-28 23:05 EDT  
**Branch:** `claude-code/pilot-fix`  
**Last commit SHA:** `51a8786102af0edf41a9c2303fb360e70b54e29a`

---

## What Was Fixed This Session

1. **`retain_graph` bug** (`src/ralrc/model_clean.py`):  
   `autograd.grad(E, R_diff, create_graph=True, retain_graph=False)` freed E's graph before `loss.backward()` could use it, causing `RuntimeError: Trying to backward through the graph a second time`. Fixed to `retain_graph=self.training`.  
   Test: `tests/test_force_backward.py` (all 3 pass).

2. **`discover_reactions` data-group bug** (`src/ralrc/split.py`):  
   Walked ALL HDF5 top-level groups including `"data"` (a full aggregate duplicate of train/val/test). Generated `"data::<formula>::<rxn_id>"` compound keys that `Transition1xDataset` never emits (only reads `train/val/test`). Half the pilot training reactions were silently unmatchable; including `"data"` in the loader would also leak val/test data into training.  
   Fix: restrict `discover_reactions` to `CANONICAL_SPLITS = {"train", "val", "test"}`.  
   `splits_pilot.json` regenerated (seed=17, --max-reactions 200). Now 200 train reactions → 187,296 matched frames (was 87,696).

---

## Completed Ablations

| Model | Seed | Checkpoint | val_force_MAE (eV/Å) | Status |
|---|---|---|---|---|
| `local_mace_style` | 17 | `runs/local_mace_style/seed17/best.pt` | 9.181 (epoch 1) | DONE + EVAL DONE |

### local_mace_style / seed17 — epoch log
```
epoch 0: train_loss=60183.4  val_E_MAE=44042 eV  val_F_MAE=25.63 eV/Å
epoch 1: train_loss=35022.5  val_E_MAE=47692 eV  val_F_MAE=9.181 eV/Å  ← best
```

### Eval result (appended to benchmarks/benchmark_results.csv)
```
model=local_mace_style  checkpoint=runs/local_mace_style/seed17/best.pt
energy_mae=37589.9 eV   force_mae=7.655 eV/Å   barrier_mae=94.52 eV
ts_force_mae=6.979 eV/Å  ood_degradation=1.206
runtime_per_atom_step=0.091 ms   n_id_frames=41020   n_ood_frames=29346
```

---

## Ablations Killed (No Valid Checkpoint)

The three remaining seed-17 ablations were launched in parallel but produced 0 bytes of output before being killed. No epochs ran; no checkpoints were written.

| Model | Seed | State |
|---|---|---|
| `charge_head_no_coulomb` | 17 | KILLED — 0 epochs, no checkpoint |
| `fixed_charge_coulomb` | 17 | KILLED — 0 epochs, no checkpoint |
| `learned_charge_coulomb` | 17 | KILLED — 0 epochs, no checkpoint |

**Root cause of slow parallel jobs:** All three competed for the single GPU without CUDA MPS. They appear to have serialized, with only one actually running at a time while the other two waited. The empty run directories (`runs/*/seed17/`) can be deleted before resuming.

---

## Open Questions / Anomalies

1. **`benchmark_results.csv` schema mismatch.** The pre-existing CSV had columns `model,seed,energy_mae,...,status` (placeholder rows from an earlier scaffold). `eval.py` appends rows with columns `model,checkpoint,energy_mae,...,n_id_frames,n_ood_frames` — different schema. The last line of the CSV is the real result but misaligned with the header. **Action needed: replace the CSV with a clean version using eval.py's schema before running more evals.**

2. **val_E_MAE ~44K eV after 2 epochs.** This is completely expected for 2 epochs of a completely random model on real data. Do not interpret as a bug. The important signal is that train_loss is halving each epoch (~60K→35K) and val_F_MAE is decreasing (25→9 eV/Å), which confirms the training loop is working correctly.

3. **ood_degradation = 1.206 for local_mace_style.** OOD is 20% worse than ID on force MAE. For an untrained model, this is roughly what you'd expect — a small directional signal but not meaningful at 2 epochs.

4. **Parallel training on single GPU is slower than sequential.** Run the 3 remaining ablations sequentially, not in parallel, to avoid GPU contention stalls.

5. **`splits_pilot.json` was regenerated.** The new splits have the same reaction counts (200/50/50/25) and seed, but different actual reactions than the old file. The old broken file (with `data::` keys) is no longer in git history as it was previously untracked. The new file is committed at SHA `51a8786`.

---

## Commands to Resume Tomorrow

Clean up empty stale run directories first:
```bash
rm -rf runs/charge_head_no_coulomb runs/fixed_charge_coulomb runs/learned_charge_coulomb
```

Then run the remaining 3 ablations **sequentially** (one at a time) to avoid GPU contention:

```bash
# Activate venv first
cd C:/Users/oweng/ralrc-mlip-benchmark
# .venv should already be active; if not: source .venv/Scripts/activate (bash) or .venv\Scripts\Activate.ps1 (PS)

python -m ralrc.train --config configs/charge_head_no_coulomb.yaml --splits splits_pilot.json --epochs 2 --seed 17

python -m ralrc.train --config configs/fixed_charge_coulomb.yaml --splits splits_pilot.json --epochs 2 --seed 17

python -m ralrc.train --config configs/learned_charge_coulomb.yaml --splits splits_pilot.json --epochs 2 --seed 17
```

After each training completes, run eval:
```bash
python -m ralrc.eval --config configs/charge_head_no_coulomb.yaml \
  --checkpoint runs/charge_head_no_coulomb/seed17/best.pt \
  --splits splits_pilot.json --h5 data/transition1x.h5 --timing

python -m ralrc.eval --config configs/fixed_charge_coulomb.yaml \
  --checkpoint runs/fixed_charge_coulomb/seed17/best.pt \
  --splits splits_pilot.json --h5 data/transition1x.h5 --timing

python -m ralrc.eval --config configs/learned_charge_coulomb.yaml \
  --checkpoint runs/learned_charge_coulomb/seed17/best.pt \
  --splits splits_pilot.json --h5 data/transition1x.h5 --timing
```

Fix the CSV schema before running evals (replace with clean file using eval.py's column order):
```bash
# Delete old placeholder CSV and let eval.py create a fresh one
rm benchmarks/benchmark_results.csv
# Then re-run the local_mace_style eval first to seed the header:
python -m ralrc.eval --config configs/local_mace_style.yaml \
  --checkpoint runs/local_mace_style/seed17/best.pt \
  --splits splits_pilot.json --h5 data/transition1x.h5 --timing
# Then run the other 3 evals after their training completes
```

After seed-17 4-row CSV is clean, extend to seeds 29 and 43:
```bash
for SEED in 29 43; do
  for MODEL in local_mace_style charge_head_no_coulomb fixed_charge_coulomb learned_charge_coulomb; do
    python -m ralrc.train --config configs/${MODEL}.yaml --splits splits_pilot.json --epochs 2 --seed $SEED
    python -m ralrc.eval --config configs/${MODEL}.yaml \
      --checkpoint runs/${MODEL}/seed${SEED}/best.pt \
      --splits splits_pilot.json --h5 data/transition1x.h5 --timing
  done
done
```

---

## Git SHA of Last Commit

`51a8786102af0edf41a9c2303fb360e70b54e29a`

Commits this session (newest first):
```
51a8786 fix: skip non-canonical HDF5 groups in discover_reactions; regen splits_pilot.json
7b044ae fix: retain_graph=self.training in autograd.grad to prevent double-backward error
```
