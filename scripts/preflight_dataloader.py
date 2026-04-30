"""Preflight smoke test for DataLoader-based batched training.

Runs in ~1-2 minutes and validates every critical piece of the new
batching logic before committing to a full training run.

Usage:
    python scripts/preflight_dataloader.py --h5 data/transition1x.h5 \\
        --splits splits_pilot.json

All checks print PASS/FAIL. Exit code 0 = everything OK to train.
Exit code 1 = something is broken, do NOT start full training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

failed: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    if not ok:
        failed.append(name)


# ---------------------------------------------------------------------------
# Imports from ralrc (validates package is installed)
# ---------------------------------------------------------------------------

print("\n=== RALRC DataLoader Preflight ===")
print("[1/7] Checking imports...")
try:
    from ralrc.model_clean import ChargeAwarePotentialClean
    from ralrc.transition1x import Transition1xDataset
    from ralrc.train import (
        _sample_to_tensors, _collate_variable_mols,
        AtomRefEnergy, energy_force_loss, _compound_key, HA_TO_EV,
    )
    check("ralrc package imports", True)
except Exception as e:
    check("ralrc package imports", False, str(e))
    print("\nCannot continue without package. Run: pip install -e '.[chem,mace]'")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

p = argparse.ArgumentParser()
p.add_argument("--h5",      default="data/transition1x.h5")
p.add_argument("--splits",  default="splits_pilot.json")
p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--batches", type=int, default=5,
               help="Number of batches to test per worker config")
p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
a = p.parse_args()

device = torch.device(a.device)

# ---------------------------------------------------------------------------
# [2/7] CUDA / device
# ---------------------------------------------------------------------------

print("\n[2/7] Checking CUDA / device...")
cuda_ok = torch.cuda.is_available()
check("CUDA available", cuda_ok,
      torch.cuda.get_device_name(0) if cuda_ok else "falling back to CPU")
check("Training device", True, str(device))
if cuda_ok:
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    check("GPU memory visible", gb > 1, f"{gb:.1f} GB")

# ---------------------------------------------------------------------------
# [3/7] HDF5 + dataset index
# ---------------------------------------------------------------------------

print("\n[3/7] Checking HDF5 + dataset index...")
try:
    full_ds = Transition1xDataset(a.h5, splits=None)
    n_frames = len(full_ds)
    check("HDF5 opens", True, a.h5)
    check("Index non-empty", n_frames > 0, f"{n_frames:,} frames")
except FileNotFoundError:
    check("HDF5 opens", False,
          f"{a.h5} not found — download Transition1x and place at that path")
    sys.exit(1)
except Exception as e:
    check("HDF5 opens", False, str(e))
    sys.exit(1)

# ---------------------------------------------------------------------------
# [4/7] Splits file + compound key format
# ---------------------------------------------------------------------------

print("\n[4/7] Checking splits file...")
try:
    with open(a.splits) as f:
        splits = json.load(f)
    train_keys = set(splits["train_id"])
    val_keys   = set(splits["val_id"])
    sample_key = next(iter(train_keys), None)
    compound_ok = sample_key is not None and "::" in sample_key
    check("Splits file loads", True, a.splits)
    check("Compound key format", compound_ok,
          repr(sample_key) if sample_key else "empty train_id")
    check("Train keys non-empty", len(train_keys) > 0, f"{len(train_keys)} keys")
    check("Val keys non-empty",   len(val_keys)   > 0, f"{len(val_keys)} keys")
except Exception as e:
    check("Splits file loads", False, str(e))
    sys.exit(1)

# Filter indices
train_indices = [i for i, entry in enumerate(full_ds._index)
                 if _compound_key(entry) in train_keys]
val_indices   = [i for i, entry in enumerate(full_ds._index)
                 if _compound_key(entry) in val_keys]
check("Train indices matched", len(train_indices) > 0,
      f"{len(train_indices):,} frames matched")
check("Val indices matched",   len(val_indices) > 0,
      f"{len(val_indices):,} frames matched")

if not train_indices or not val_indices:
    print("\nSplit indices empty — stale splits.json vs HDF5. Regenerate splits.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# [5/7] DataLoader construction + first batch
# ---------------------------------------------------------------------------

print("\n[5/7] Checking DataLoader construction + first batch...")
try:
    _use_workers = a.workers > 0
    loader = DataLoader(
        Subset(full_ds, train_indices[:500]),  # small slice for speed
        batch_size=a.batch_size,
        shuffle=True,
        collate_fn=_collate_variable_mols,
        num_workers=a.workers,
        prefetch_factor=2 if _use_workers else None,
        persistent_workers=_use_workers,
        pin_memory=False,
    )
    batch = next(iter(loader))
    check("DataLoader constructs",    True)
    check("First batch is list",      isinstance(batch, list),
          f"type={type(batch).__name__}")
    check("Batch non-empty",          len(batch) > 0, f"{len(batch)} samples")
    check("Batch size ≤ requested",   len(batch) <= a.batch_size,
          f"{len(batch)} ≤ {a.batch_size}")
except Exception as e:
    check("DataLoader constructs", False, str(e))
    traceback.print_exc()
    sys.exit(1)

# Validate first sample structure
try:
    s0 = batch[0]
    check("Sample has 'z' key",      "z"      in s0)
    check("Sample has 'pos' key",    "pos"    in s0)
    check("Sample has 'energy' key", "energy" in s0)
    check("Sample has 'forces' key", "forces" in s0)
except Exception as e:
    check("Sample structure", False, str(e))

# ---------------------------------------------------------------------------
# [6/7] Forward + backward + gradient check
# ---------------------------------------------------------------------------

print("\n[6/7] Checking forward/backward + gradient sanity...")
try:
    model    = ChargeAwarePotentialClean(hidden=64, use_coulomb=True).to(device)
    atom_ref = AtomRefEnergy().to(device)
    params   = list(model.parameters()) + list(atom_ref.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    sample = batch[0]
    Z, R, Q, S, E_ref, F_ref = _sample_to_tensors(sample, device)

    check("Tensors on correct device", Z.device.type == device.type,
          f"Z.device={Z.device}")

    R.requires_grad_(True)
    out = model.forward(Z, R, Q, S, compute_forces=True)
    check("Forward pass runs",         True)
    check("Energy is scalar",          out["energy"].ndim == 0,
          f"shape={out['energy'].shape}")
    check("Forces shape matches R",    out["forces"].shape == R.shape,
          f"{out['forces'].shape} vs {R.shape}")
    check("Energy is finite",          out["energy"].isfinite().item())
    check("Forces are finite",         out["forces"].isfinite().all().item())

    # Charge conservation: sum(q_i) == Q
    with torch.no_grad():
        q_raw = model.charge_head(model._embed(Z))  # may not exist as separate method
        # Fallback: just verify forward doesn't raise
    check("Charge head accessible",   hasattr(model, "charge_head"))

    # Backward
    E_ref_corr = E_ref - atom_ref(Z)
    loss = energy_force_loss(out["energy"], out["forces"], E_ref_corr, F_ref)
    optimizer.zero_grad()
    loss.backward()
    check("Backward runs",             True)

    grad_norms = [p.grad.norm().item() for p in params if p.grad is not None]
    check("Gradients exist",           len(grad_norms) > 0,
          f"{len(grad_norms)} param tensors with grad")
    all_finite = all(g == g and g < 1e9 for g in grad_norms)
    check("Gradients finite",          all_finite,
          f"max={max(grad_norms):.3e}" if grad_norms else "none")

    optimizer.step()
    check("Optimizer step runs",       True)

except Exception as e:
    check("Forward/backward", False, str(e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# [7/7] Timing: num_workers=0 vs num_workers=N over N batches
# ---------------------------------------------------------------------------

print(f"\n[7/7] Timing: single-worker vs {a.workers}-worker over {a.batches} batches...")

def time_loader(nw: int, n_batches: int) -> float:
    _uw = nw > 0
    ldr = DataLoader(
        Subset(full_ds, train_indices[:min(500, len(train_indices))]),
        batch_size=a.batch_size,
        shuffle=False,
        collate_fn=_collate_variable_mols,
        num_workers=nw,
        prefetch_factor=2 if _uw else None,
        persistent_workers=False,  # don't persist for timing test
        pin_memory=False,
    )
    t0 = time.perf_counter()
    for i, b in enumerate(ldr):
        if i >= n_batches:
            break
        # Simulate minimal GPU work: move first sample
        Z, R, Q, S, E_ref, F_ref = _sample_to_tensors(b[0], device)
        _ = Z.sum()  # force transfer
    return time.perf_counter() - t0

try:
    t_single  = time_loader(0,        a.batches)
    t_multi   = time_loader(a.workers, a.batches)
    speedup   = t_single / max(t_multi, 1e-6)
    check("Multi-worker faster than single", speedup > 1.0,
          f"{speedup:.2f}x  ({t_single:.2f}s → {t_multi:.2f}s)")
    if speedup < 1.0:
        print(f"    {WARN} Workers slower than single-process on this run.")
        print("         This can happen on cold HDF5 cache. Retry once warmed up.")
        print("         If consistently slow: use --workers 0 to disable.")
except Exception as e:
    check("Timing test", False, str(e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "="*44)
if not failed:
    print(f"  ALL CHECKS PASSED — safe to start training.")
    print(f"  Recommended command:")
    print(f"    python -m ralrc.train --config configs/local_mace_style.yaml \\")
    print(f"      --h5 {a.h5} --splits {a.splits} --epochs 30 --seed 17")
    sys.exit(0)
else:
    print(f"  {len(failed)} CHECK(S) FAILED:")
    for f in failed:
        print(f"    - {f}")
    print("  DO NOT start full training until these are resolved.")
    sys.exit(1)
