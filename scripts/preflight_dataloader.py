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
# Module-level helpers only - no logic, no I/O, no DataLoader calls here.
# Workers import this module and must not trigger any side effects.
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
INFO = "\033[94mINFO\033[0m"


def check(name: str, ok: bool, detail: str = "", _failed_list: list = None):
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    if not ok and _failed_list is not None:
        _failed_list.append(name)


def warn(name: str, detail: str = ""):
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{WARN}] {name}{suffix}")


def info(name: str, detail: str = ""):
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{INFO}] {name}{suffix}")


def _make_timing_loader(full_ds, indices, batch_size, nw, collate_fn):
    """Build a persistent DataLoader for timing. Must be called inside __main__."""
    from torch.utils.data import DataLoader, Subset
    _uw = nw > 0
    return DataLoader(
        Subset(full_ds, indices[:min(1000, len(indices))]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=nw,
        prefetch_factor=2 if _uw else None,
        persistent_workers=_uw,   # keep workers alive - mirrors real training
        pin_memory=False,
    )


def _drain_loader(loader, n_batches, device, sample_to_tensors):
    """Consume n_batches from loader, returning wall time (excludes first warmup)."""
    import torch
    it = iter(loader)
    # Warmup: one batch to prime HDF5 page cache and worker pipes
    try:
        warmup = next(it)
        Z, *_ = sample_to_tensors(warmup[0], device)
        _ = Z.sum().item()  # force GPU sync
    except StopIteration:
        return 0.0

    t0 = time.perf_counter()
    count = 0
    for b in it:
        Z, *_ = sample_to_tensors(b[0], device)
        _ = Z.sum().item()
        count += 1
        if count >= n_batches:
            break
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# ALL logic lives here - safe for Windows spawn workers to import this file
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    multiprocessing.freeze_support()  # required for Windows; no-op otherwise

    failed: list[str] = []
    recommended_workers: int = 4  # updated by Check 7

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
    p.add_argument("--h5",         default="data/transition1x.h5")
    p.add_argument("--splits",     default="splits_pilot.json")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--batches",    type=int, default=20,
                   help="Batches to time after warmup (default 20)")
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
        check("HDF5 opens",     True, a.h5,                   _failed_list=failed)
        check("Index non-empty", n_frames > 0,
              f"{n_frames:,} frames",                          _failed_list=failed)
    except FileNotFoundError:
        check("HDF5 opens", False,
              f"{a.h5} not found - download Transition1x first",
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
            splits_data = json.load(f)
        train_keys = set(splits_data["train_id"])
        val_keys   = set(splits_data["val_id"])
        sample_key = next(iter(train_keys), None)
        compound_ok = sample_key is not None and "::" in sample_key
        check("Splits file loads",    True, a.splits,         _failed_list=failed)
        check("Compound key format",  compound_ok,
              repr(sample_key) if sample_key else "empty",    _failed_list=failed)
        check("Train keys non-empty", len(train_keys) > 0,
              f"{len(train_keys)} keys",                      _failed_list=failed)
        check("Val keys non-empty",   len(val_keys) > 0,
              f"{len(val_keys)} keys",                        _failed_list=failed)
    except Exception as e:
        check("Splits file loads", False, str(e), _failed_list=failed)
        sys.exit(1)

    train_indices = [i for i, entry in enumerate(full_ds._index)
                     if _compound_key(entry) in train_keys]
    val_indices   = [i for i, entry in enumerate(full_ds._index)
                     if _compound_key(entry) in val_keys]
    check("Train indices matched", len(train_indices) > 0,
          f"{len(train_indices):,} frames",                   _failed_list=failed)
    check("Val indices matched",   len(val_indices) > 0,
          f"{len(val_indices):,} frames",                     _failed_list=failed)

    if not train_indices or not val_indices:
        print("\nSplit indices empty - stale splits.json vs HDF5.")
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
        check("DataLoader constructs",   True,                _failed_list=failed)
        check("First batch is list",     isinstance(batch, list),
              f"type={type(batch).__name__}",                  _failed_list=failed)
        check("Batch non-empty",         len(batch) > 0,
              f"{len(batch)} samples",                         _failed_list=failed)
        check("Batch size <= requested", len(batch) <= a.batch_size,
              f"{len(batch)} <= {a.batch_size}",              _failed_list=failed)
    except Exception as e:
        check("DataLoader constructs", False, str(e),         _failed_list=failed)
        traceback.print_exc()
        print("\nDataLoader failed. Check 7 will use workers=0.")
        a.workers = 0
        recommended_workers = 0

    if batch is not None:
        s0 = batch[0]
        check("Sample has 'z' key",      "z"      in s0,      _failed_list=failed)
        check("Sample has 'pos' key",    "pos"    in s0,      _failed_list=failed)
        check("Sample has 'energy' key", "energy" in s0,      _failed_list=failed)
        check("Sample has 'forces' key", "forces" in s0,      _failed_list=failed)

    # -----------------------------------------------------------------------
    # [6/7] Forward + backward + gradient check
    # -----------------------------------------------------------------------
    print("\n[6/7] Checking forward/backward + gradient sanity...")
    if batch is None:
        batch = [full_ds[train_indices[0]]]
    try:
        model    = ChargeAwarePotentialClean(hidden=64, use_coulomb=True).to(device)
        atom_ref = AtomRefEnergy().to(device)
        params   = list(model.parameters()) + list(atom_ref.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        sample = batch[0]
        Z, R, Q, S, E_ref, F_ref = _sample_to_tensors(sample, device)

        check("Tensors on correct device", Z.device.type == device.type,
              f"Z.device={Z.device}",                         _failed_list=failed)

        R.requires_grad_(True)
        out = model.forward(Z, R, Q, S, compute_forces=True)
        check("Forward pass runs",  True,                     _failed_list=failed)
        check("Energy is scalar",   out["energy"].ndim == 0,
              f"shape={out['energy'].shape}",                 _failed_list=failed)
        check("Forces shape == R",  out["forces"].shape == R.shape,
              f"{out['forces'].shape} vs {R.shape}",          _failed_list=failed)
        check("Energy is finite",   out["energy"].isfinite().item(),
              _failed_list=failed)
        check("Forces are finite",  out["forces"].isfinite().all().item(),
              _failed_list=failed)
        check("Charge head exists", hasattr(model, "charge_head"),
              _failed_list=failed)

        E_ref_corr = E_ref - atom_ref(Z)
        loss = energy_force_loss(out["energy"], out["forces"], E_ref_corr, F_ref)
        optimizer.zero_grad()
        loss.backward()
        check("Backward runs",    True,                       _failed_list=failed)

        grad_norms = [p.grad.norm().item() for p in params if p.grad is not None]
        check("Gradients exist",  len(grad_norms) > 0,
              f"{len(grad_norms)} tensors",                   _failed_list=failed)
        all_finite = all(g == g and g < 1e9 for g in grad_norms)
        check("Gradients finite", all_finite,
              f"max={max(grad_norms):.3e}" if grad_norms else "none",
              _failed_list=failed)
        optimizer.step()
        check("Optimizer step",   True,                       _failed_list=failed)

    except Exception as e:
        check("Forward/backward", False, str(e),              _failed_list=failed)
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # [7/7] Timing (informational WARN only - never a hard FAIL)
    #
    # Why timing is advisory:
    #   On Windows, worker spawn costs ~4s per process. With only a handful
    #   of batches and persistent_workers=False, spawn dominates and workers
    #   look slower. In real training, persistent_workers=True amortizes the
    #   spawn cost across all epochs (~5850 batches/epoch here), so workers
    #   are always a net win. We test with persistent_workers=True + warmup
    #   to get a realistic signal, but even a negative result here just means
    #   use --num-workers 0; it does NOT block training correctness.
    # -----------------------------------------------------------------------
    print(f"\n[7/7] Timing benchmark (informational - will not block training)...")
    print(f"      persistent_workers=True, 1 warmup batch + {a.batches} timed batches")
    try:
        ldr_single = _make_timing_loader(full_ds, train_indices,
                                         a.batch_size, 0, _collate_variable_mols)
        ldr_multi  = _make_timing_loader(full_ds, train_indices,
                                         a.batch_size, a.workers,
                                         _collate_variable_mols)

        t_single = _drain_loader(ldr_single, a.batches, device, _sample_to_tensors)
        t_multi  = _drain_loader(ldr_multi,  a.batches, device, _sample_to_tensors)
        speedup  = t_single / max(t_multi, 1e-6)

        if speedup >= 1.0:
            info(f"{a.workers}-worker speedup",
                 f"{speedup:.2f}x  ({t_single:.3f}s -> {t_multi:.3f}s) - workers recommended")
            recommended_workers = a.workers
        else:
            warn(f"{a.workers}-worker NOT faster after warmup",
                 f"{speedup:.2f}x  ({t_single:.3f}s -> {t_multi:.3f}s)")
            print(f"      This means your HDF5 reads are fast enough that worker")
            print(f"      overhead exceeds prefetch benefit for this molecule size.")
            print(f"      Training will still work correctly with --num-workers 0.")
            recommended_workers = 0

    except Exception as e:
        warn("Timing test error", str(e))
        traceback.print_exc()
        recommended_workers = 0

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 44)
    if not failed:
        print("  ALL CHECKS PASSED - safe to start training.")
        print(f"  Recommended num_workers for your system: {recommended_workers}")
        print()
        print("  Training commands:")
        for cfg in [
            "configs/local_mace_style.yaml",
            "configs/charge_head_no_coulomb.yaml",
            "configs/fixed_charge_coulomb.yaml",
            "configs/learned_charge_coulomb.yaml",
        ]:
            print(f"    python -m ralrc.train --config {cfg} \\")
            print(f"      --h5 {a.h5} --splits {a.splits} \\"
                  f" --epochs 30 --seed 17 --num-workers {recommended_workers}")
            print()
        sys.exit(0)
    else:
        print(f"  {len(failed)} CHECK(S) FAILED:")
        for name in failed:
            print(f"    - {name}")
        print("  DO NOT start full training until resolved.")
        sys.exit(1)
