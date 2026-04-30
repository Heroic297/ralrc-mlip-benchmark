"""Preflight smoke test for DataLoader-based batched training.

Runs in ~1-2 minutes and validates every critical piece of the new
batching logic before committing to a full training run.

Usage:
    python scripts/preflight_dataloader.py --h5 data/transition1x.h5 \\
        --splits splits_pilot.json

All checks print PASS/FAIL. Exit code 0 = everything OK to train.
Exit code 1 = something is broken, do NOT start full training.

Windows note: all logic is inside if __name__ == '__main__' so that
the Windows 'spawn' multiprocessing backend does not re-execute
top-level code when it imports this module into worker processes.
"""
from __future__ import annotations

import multiprocessing
import argparse
import json
import sys
import time
import traceback


# ---------------------------------------------------------------------------
# Module-level helpers only — no logic, no I/O, no DataLoader calls here.
# Workers import this module and must not trigger any side effects.
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def check(name: str, ok: bool, detail: str = "", _failed_list: list = None):
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    if not ok and _failed_list is not None:
        _failed_list.append(name)


def time_loader(full_ds, train_indices, batch_size, device, nw, n_batches,
               collate_fn, sample_to_tensors):
    """Time N batches of a DataLoader with nw workers."""
    import torch
    from torch.utils.data import DataLoader, Subset
    _uw = nw > 0
    ldr = DataLoader(
        Subset(full_ds, train_indices[:min(500, len(train_indices))]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=nw,
        prefetch_factor=2 if _uw else None,
        persistent_workers=False,
        pin_memory=False,
    )
    t0 = time.perf_counter()
    for i, b in enumerate(ldr):
        if i >= n_batches:
            break
        Z, R, Q, S, E_ref, F_ref = sample_to_tensors(b[0], device)
        _ = Z.sum()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# ALL logic lives here — safe for Windows spawn workers to import this file
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    multiprocessing.freeze_support()  # no-op on non-frozen, required on Windows

    failed: list[str] = []

    # -----------------------------------------------------------------------
    # [1/7] Imports
    # -----------------------------------------------------------------------
    print("\n=== RALRC DataLoader Preflight ===")
    print("[1/7] Checking imports...")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Subset
        from ralrc.model_clean import ChargeAwarePotentialClean
        from ralrc.transition1x import Transition1xDataset
        from ralrc.train import (
            _sample_to_tensors, _collate_variable_mols,
            AtomRefEnergy, energy_force_loss, _compound_key,
        )
        check("ralrc package imports", True, _failed_list=failed)
    except Exception as e:
        check("ralrc package imports", False, str(e), _failed_list=failed)
        print("\nCannot continue. Run: pip install -e '.[chem]'")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # CLI
    # -----------------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("--h5",        default="data/transition1x.h5")
    p.add_argument("--splits",    default="splits_pilot.json")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--batches",   type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    a = p.parse_args()
    device = torch.device(a.device)

    # -----------------------------------------------------------------------
    # [2/7] CUDA / device
    # -----------------------------------------------------------------------
    print("\n[2/7] Checking CUDA / device...")
    cuda_ok = torch.cuda.is_available()
    check("CUDA available", cuda_ok,
          torch.cuda.get_device_name(0) if cuda_ok else "falling back to CPU",
          _failed_list=failed)
    check("Training device", True, str(device), _failed_list=failed)
    if cuda_ok:
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        check("GPU memory visible", gb > 1, f"{gb:.1f} GB", _failed_list=failed)

    # -----------------------------------------------------------------------
    # [3/7] HDF5 + dataset index
    # -----------------------------------------------------------------------
    print("\n[3/7] Checking HDF5 + dataset index...")
    try:
        full_ds = Transition1xDataset(a.h5, splits=None)
        n_frames = len(full_ds)
        check("HDF5 opens",    True, a.h5,                    _failed_list=failed)
        check("Index non-empty", n_frames > 0, f"{n_frames:,} frames", _failed_list=failed)
    except FileNotFoundError:
        check("HDF5 opens", False,
              f"{a.h5} not found — download Transition1x first",
              _failed_list=failed)
        sys.exit(1)
    except Exception as e:
        check("HDF5 opens", False, str(e), _failed_list=failed)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # [4/7] Splits file + compound key format
    # -----------------------------------------------------------------------
    print("\n[4/7] Checking splits file...")
    try:
        with open(a.splits) as f:
            splits = json.load(f)
        train_keys = set(splits["train_id"])
        val_keys   = set(splits["val_id"])
        sample_key = next(iter(train_keys), None)
        compound_ok = sample_key is not None and "::" in sample_key
        check("Splits file loads",   True, a.splits,          _failed_list=failed)
        check("Compound key format", compound_ok,
              repr(sample_key) if sample_key else "empty",     _failed_list=failed)
        check("Train keys non-empty", len(train_keys) > 0,
              f"{len(train_keys)} keys",                       _failed_list=failed)
        check("Val keys non-empty",   len(val_keys) > 0,
              f"{len(val_keys)} keys",                         _failed_list=failed)
    except Exception as e:
        check("Splits file loads", False, str(e), _failed_list=failed)
        sys.exit(1)

    train_indices = [i for i, entry in enumerate(full_ds._index)
                     if _compound_key(entry) in train_keys]
    val_indices   = [i for i, entry in enumerate(full_ds._index)
                     if _compound_key(entry) in val_keys]
    check("Train indices matched", len(train_indices) > 0,
          f"{len(train_indices):,} frames",                    _failed_list=failed)
    check("Val indices matched",   len(val_indices) > 0,
          f"{len(val_indices):,} frames",                      _failed_list=failed)

    if not train_indices or not val_indices:
        print("\nSplit indices empty — stale splits.json vs HDF5.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # [5/7] DataLoader construction + first batch
    # -----------------------------------------------------------------------
    print("\n[5/7] Checking DataLoader construction + first batch...")
    batch = None
    try:
        _use_workers = a.workers > 0
        loader = DataLoader(
            Subset(full_ds, train_indices[:500]),
            batch_size=a.batch_size,
            shuffle=True,
            collate_fn=_collate_variable_mols,
            num_workers=a.workers,
            prefetch_factor=2 if _use_workers else None,
            persistent_workers=_use_workers,
            pin_memory=False,
        )
        batch = next(iter(loader))
        check("DataLoader constructs",  True,                  _failed_list=failed)
        check("First batch is list",    isinstance(batch, list),
              f"type={type(batch).__name__}",                   _failed_list=failed)
        check("Batch non-empty",        len(batch) > 0,
              f"{len(batch)} samples",                          _failed_list=failed)
        check("Batch size <= requested", len(batch) <= a.batch_size,
              f"{len(batch)} <= {a.batch_size}",               _failed_list=failed)
    except Exception as e:
        check("DataLoader constructs", False, str(e),          _failed_list=failed)
        traceback.print_exc()
        print("\nDataLoader failed. Falling back to --workers 0 in timing test.")
        a.workers = 0

    if batch is not None:
        s0 = batch[0]
        check("Sample has 'z' key",      "z"      in s0,       _failed_list=failed)
        check("Sample has 'pos' key",    "pos"    in s0,       _failed_list=failed)
        check("Sample has 'energy' key", "energy" in s0,       _failed_list=failed)
        check("Sample has 'forces' key", "forces" in s0,       _failed_list=failed)

    # -----------------------------------------------------------------------
    # [6/7] Forward + backward + gradient check
    # -----------------------------------------------------------------------
    print("\n[6/7] Checking forward/backward + gradient sanity...")
    if batch is None:
        # Fall back to loading one sample directly
        batch = [full_ds[train_indices[0]]]
    try:
        model    = ChargeAwarePotentialClean(hidden=64, use_coulomb=True).to(device)
        atom_ref = AtomRefEnergy().to(device)
        params   = list(model.parameters()) + list(atom_ref.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        sample = batch[0]
        Z, R, Q, S, E_ref, F_ref = _sample_to_tensors(sample, device)

        check("Tensors on correct device", Z.device.type == device.type,
              f"Z.device={Z.device}",                          _failed_list=failed)

        R.requires_grad_(True)
        out = model.forward(Z, R, Q, S, compute_forces=True)
        check("Forward pass runs",   True,                     _failed_list=failed)
        check("Energy is scalar",    out["energy"].ndim == 0,
              f"shape={out['energy'].shape}",                  _failed_list=failed)
        check("Forces shape == R",   out["forces"].shape == R.shape,
              f"{out['forces'].shape} vs {R.shape}",           _failed_list=failed)
        check("Energy is finite",    out["energy"].isfinite().item(),
              _failed_list=failed)
        check("Forces are finite",   out["forces"].isfinite().all().item(),
              _failed_list=failed)
        check("Charge head exists",  hasattr(model, "charge_head"),
              _failed_list=failed)

        E_ref_corr = E_ref - atom_ref(Z)
        loss = energy_force_loss(out["energy"], out["forces"], E_ref_corr, F_ref)
        optimizer.zero_grad()
        loss.backward()
        check("Backward runs", True,                           _failed_list=failed)

        grad_norms = [p.grad.norm().item() for p in params if p.grad is not None]
        check("Gradients exist",  len(grad_norms) > 0,
              f"{len(grad_norms)} tensors",                    _failed_list=failed)
        all_finite = all(g == g and g < 1e9 for g in grad_norms)
        check("Gradients finite", all_finite,
              f"max={max(grad_norms):.3e}" if grad_norms else "none",
              _failed_list=failed)

        optimizer.step()
        check("Optimizer step", True,                          _failed_list=failed)

    except Exception as e:
        check("Forward/backward", False, str(e),               _failed_list=failed)
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # [7/7] Timing: 0 workers vs N workers
    # -----------------------------------------------------------------------
    print(f"\n[7/7] Timing: single-worker vs {a.workers}-worker "
          f"over {a.batches} batches...")
    try:
        t_single = time_loader(full_ds, train_indices, a.batch_size, device,
                               0, a.batches, _collate_variable_mols,
                               _sample_to_tensors)
        t_multi  = time_loader(full_ds, train_indices, a.batch_size, device,
                               a.workers, a.batches, _collate_variable_mols,
                               _sample_to_tensors)
        speedup  = t_single / max(t_multi, 1e-6)
        check("Multi-worker faster", speedup > 1.0,
              f"{speedup:.2f}x  ({t_single:.2f}s -> {t_multi:.2f}s)",
              _failed_list=failed)
        if speedup < 1.0:
            print(f"    [{WARN}] Workers slower this run (cold cache).")
            print("            Retry once warmed up. If still slow, use --workers 0.")
    except Exception as e:
        check("Timing test", False, str(e), _failed_list=failed)
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 44)
    if not failed:
        print("  ALL CHECKS PASSED - safe to start training.")
        print("  Recommended command:")
        print(f"    python -m ralrc.train --config configs/local_mace_style.yaml ")
        print(f"      --h5 {a.h5} --splits {a.splits} --epochs 30 --seed 17")
        sys.exit(0)
    else:
        print(f"  {len(failed)} CHECK(S) FAILED:")
        for name in failed:
            print(f"    - {name}")
        print("  DO NOT start full training until resolved.")
        sys.exit(1)
