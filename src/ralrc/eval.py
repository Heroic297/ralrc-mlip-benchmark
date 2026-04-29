"""Full evaluation harness for RALRC ablation models.

Metrics written to benchmarks/benchmark_results.csv:
  energy_mae          eV/molecule
  force_mae           eV/Å
  barrier_mae         eV  (max-E minus min-E per reaction IRC)
  ts_force_mae        eV/Å  (TS-neighbourhood frames only)
  ood_degradation     force_MAE_OOD / force_MAE_ID
  runtime_per_atom_step  ms/(atom*step)

Split membership uses COMPOUND KEYS: "<hdf5_split>::<formula>::<rxn_id>"

Hard-fails if n_id_frames == 0 or n_ood_frames == 0 instead of writing nan.

Usage:
    python -m ralrc.eval --config configs/learned_charge_coulomb.yaml \\
        --checkpoint runs/learned_charge_coulomb/seed17/best.pt \\
        --h5 data/transition1x.h5 --splits splits.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from .model_clean import ChargeAwarePotentialClean
from .transition1x import Transition1xDataset
from .train import AtomRefEnergy, _sample_to_tensors, HA_TO_EV


# ---------------------------------------------------------------------------
# Compound key helper
# ---------------------------------------------------------------------------

def _compound_key(entry: tuple) -> str:
    return f"{entry[0]}::{entry[1]}::{entry[2]}"


# ---------------------------------------------------------------------------
# TS-neighbourhood detection
# ---------------------------------------------------------------------------

def _is_ts_neighbourhood(frame_idx: int, n_frames: int, window: float = 0.1) -> bool:
    if n_frames <= 0:
        return False
    mid = (n_frames - 1) / 2.0
    half_w = window * n_frames / 2.0
    return abs(frame_idx - mid) <= half_w


# ---------------------------------------------------------------------------
# Barrier estimator per reaction
# ---------------------------------------------------------------------------

def _compute_barrier(energies_ev: list[float]) -> float:
    if len(energies_ev) < 2:
        return float("nan")
    return max(energies_ev) - min(energies_ev)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: ChargeAwarePotentialClean,
    atom_ref: AtomRefEnergy,
    dataset: Transition1xDataset,
    indices: list[int],
    device: torch.device,
    ts_window: float = 0.1,
    compute_forces: bool = True,
    timing: bool = False,
) -> dict:
    model.eval()
    atom_ref.eval()

    # Group by compound key so barrier is computed per-reaction
    rxn_to_indices: dict[str, list[int]] = {}
    for i in indices:
        ck = _compound_key(dataset._index[i])
        rxn_to_indices.setdefault(ck, []).append(i)

    e_abs_errors: list[float] = []
    f_abs_errors: list[float] = []
    ts_f_errors:  list[float] = []
    barrier_errors: list[float] = []
    total_atoms = 0
    total_time_s = 0.0
    n_steps = 0

    for ck, rxn_indices in rxn_to_indices.items():
        rxn_indices_sorted = sorted(rxn_indices, key=lambda i: dataset._index[i][3])
        n_frames_rxn = len(rxn_indices_sorted)

        pred_energies: list[float] = []
        ref_energies:  list[float] = []

        for i in rxn_indices_sorted:
            sample = dataset[i]
            frame_idx = dataset._index[i][3]

            Z, R, Q, S, E_ref, F_ref = _sample_to_tensors(sample, device)
            E_ref_corr = E_ref - atom_ref(Z)

            t0 = time.perf_counter() if timing else None

            if compute_forces:
                with torch.enable_grad():
                    R_g = R.detach().requires_grad_(True)
                    out = model.forward(Z, R_g, Q, S, compute_forces=True)
            else:
                out = model.forward(Z, R, Q, S, compute_forces=False)

            if timing:
                torch.cuda.synchronize() if device.type == "cuda" else None
                total_time_s += time.perf_counter() - t0
                total_atoms += Z.shape[0]
                n_steps += 1

            E_pred = out["energy"].detach()
            e_abs_errors.append((E_pred - E_ref_corr).abs().item())

            if compute_forces and "forces" in out:
                F_pred = out["forces"].detach()
                f_err = (F_pred - F_ref).abs().mean().item()
                f_abs_errors.append(f_err)
                if _is_ts_neighbourhood(frame_idx, n_frames_rxn, ts_window):
                    ts_f_errors.append(f_err)

            pred_energies.append(E_pred.item())
            ref_energies.append(E_ref_corr.item())

        pred_barrier = _compute_barrier(pred_energies)
        ref_barrier  = _compute_barrier(ref_energies)
        if not (np.isnan(pred_barrier) or np.isnan(ref_barrier)):
            barrier_errors.append(abs(pred_barrier - ref_barrier))

    energy_mae   = float(np.mean(e_abs_errors))  if e_abs_errors   else float("nan")
    force_mae    = float(np.mean(f_abs_errors))  if f_abs_errors   else float("nan")
    barrier_mae  = float(np.mean(barrier_errors)) if barrier_errors else float("nan")
    ts_force_mae = float(np.mean(ts_f_errors))   if ts_f_errors    else float("nan")

    result = {
        "energy_mae":  energy_mae,
        "force_mae":   force_mae,
        "barrier_mae": barrier_mae,
        "ts_force_mae": ts_force_mae,
        "n_frames":    len(indices),
        "n_reactions": len(rxn_to_indices),
    }
    if timing and n_steps > 0:
        result["runtime_per_atom_step_ms"] = 1000.0 * total_time_s / max(total_atoms, 1)
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate a RALRC checkpoint.")
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5",         default=None)
    p.add_argument("--splits",     default=None)
    p.add_argument("--out",        default="benchmarks/benchmark_results.csv")
    p.add_argument("--ts-window",  type=float, default=0.1)
    p.add_argument("--timing",     action="store_true")
    p.add_argument("--device",     default=None)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    model_name = cfg.get("name", Path(a.checkpoint).parent.parent.name)

    h5_path     = a.h5     or cfg.get("h5_path",    "data/transition1x.h5")
    splits_json = a.splits or cfg.get("splits_json", "splits.json")
    device_str  = a.device or cfg.get("device",      "cuda" if torch.cuda.is_available() else "cpu")
    device      = torch.device(device_str)

    # Load checkpoint
    ckpt = torch.load(a.checkpoint, map_location=device)
    model = ChargeAwarePotentialClean(
        hidden=cfg.get("hidden", 64),
        use_coulomb=cfg.get("use_coulomb", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    atom_ref = AtomRefEnergy().to(device)
    atom_ref.load_state_dict(ckpt["atom_ref"])

    # Load split compound keys
    with open(splits_json) as f:
        splits = json.load(f)

    id_keys  = set(splits.get("test_id_same_family", []))
    ood_keys = set(splits.get("test_ood_family", []))
    val_keys = set(splits.get("val_id", []))

    # Validate that splits use compound keys
    sample_key = next(iter(id_keys), None)
    if sample_key is not None and "::" not in sample_key:
        raise ValueError(
            f"splits.json contains bare rxn_ids, not compound keys.\n"
            f"  Example: {sample_key!r}\n"
            f"  Regenerate with: python -m ralrc.split --h5 ... --out splits.json"
        )

    # Build single dataset over ALL HDF5 splits
    print("[eval] Building full dataset index (all HDF5 splits)...")
    full_ds = Transition1xDataset(h5_path, splits=None)
    print(f"[eval] Index size: {len(full_ds)} frames")

    def idx_for(key_set):
        return [i for i, entry in enumerate(full_ds._index)
                if _compound_key(entry) in key_set]

    id_indices  = idx_for(id_keys)
    ood_indices = idx_for(ood_keys)
    val_indices = idx_for(val_keys)

    print(f"[eval] {model_name}: ID={len(id_indices)} frames, "
          f"OOD={len(ood_indices)} frames, val={len(val_indices)} frames")

    # Hard-fail on empty splits
    if len(id_indices) == 0:
        raise RuntimeError(
            f"[eval] FATAL: 0 ID-test frames matched from splits.json.\n"
            f"  id_keys sample: {list(id_keys)[:3]}\n"
            f"  Ensure splits.json was generated from this HDF5 file."
        )
    if len(ood_indices) == 0:
        raise RuntimeError(
            f"[eval] FATAL: 0 OOD frames matched from splits.json.\n"
            f"  ood_keys sample: {list(ood_keys)[:3]}"
        )

    # Evaluate
    id_metrics  = evaluate(model, atom_ref, full_ds, id_indices,  device,
                           ts_window=a.ts_window, timing=a.timing)
    ood_metrics = evaluate(model, atom_ref, full_ds, ood_indices, device,
                           ts_window=a.ts_window, timing=False)

    ood_deg = float("nan")
    if id_metrics["force_mae"] > 0 and not np.isnan(ood_metrics["force_mae"]):
        ood_deg = ood_metrics["force_mae"] / id_metrics["force_mae"]

    row = {
        "model":                   model_name,
        "checkpoint":              a.checkpoint,
        "energy_mae":              round(id_metrics["energy_mae"],  6),
        "force_mae":               round(id_metrics["force_mae"],   6),
        "barrier_mae":             round(id_metrics["barrier_mae"], 6),
        "ts_force_mae":            round(id_metrics["ts_force_mae"],6),
        "ood_degradation":         round(ood_deg, 4) if not np.isnan(ood_deg) else "nan",
        "runtime_per_atom_step":   id_metrics.get("runtime_per_atom_step_ms", "nan"),
        "n_id_frames":             id_metrics["n_frames"],
        "n_ood_frames":            ood_metrics["n_frames"],
    }

    print(json.dumps(row, indent=2))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_header = not Path(a.out).exists()
    with open(a.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[eval] Results appended to {a.out}")


if __name__ == "__main__":
    main()
