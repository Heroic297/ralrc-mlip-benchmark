"""Full evaluation harness for RALRC ablation models.

Metrics computed and written to benchmarks/benchmark_results.csv:
  energy_mae          eV/molecule
  force_mae           eV/Å
  barrier_mae         eV  (max-E frame minus min-E frame per reaction)
  ts_force_mae        eV/Å  (force MAE on TS-neighbourhood frames only)
  ood_degradation     MAE_OOD / MAE_ID  (requires test_ood split)
  runtime_per_atom_step  ms/(atom*step)

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
# TS-neighbourhood detection
# ---------------------------------------------------------------------------

def _is_ts_neighbourhood(frame_idx: int, n_frames: int, window: float = 0.1) -> bool:
    """Return True if frame is within `window` fraction of the IRC midpoint."""
    if n_frames <= 0:
        return False
    mid = (n_frames - 1) / 2.0
    half_w = window * n_frames / 2.0
    return abs(frame_idx - mid) <= half_w


# ---------------------------------------------------------------------------
# Barrier estimator per reaction
# ---------------------------------------------------------------------------

def _compute_barrier(energies_ev: list[float]) -> float:
    """Max-minus-min energy along IRC as a proxy for barrier height."""
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
    """
    Evaluate model on the given index subset.

    Returns dict with keys:
        energy_mae, force_mae, barrier_mae, ts_force_mae,
        n_frames, n_reactions, timing_ms_per_atom_step (if timing=True)
    """
    model.eval()
    atom_ref.eval()

    # Group indices by reaction to compute per-reaction barriers
    rxn_to_indices: dict[str, list[int]] = {}
    for i in indices:
        rxn_id = dataset._index[i][2]
        rxn_to_indices.setdefault(rxn_id, []).append(i)

    e_abs_errors: list[float] = []
    f_abs_errors: list[float] = []  # mean per frame
    ts_f_errors: list[float] = []
    barrier_errors: list[float] = []
    total_atoms = 0
    total_time_s = 0.0
    n_steps = 0

    for rxn_id, rxn_indices in rxn_to_indices.items():
        # Sort by frame_idx so barrier is computed correctly
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

        # Per-reaction barrier MAE
        pred_barrier = _compute_barrier(pred_energies)
        ref_barrier  = _compute_barrier(ref_energies)
        if not (np.isnan(pred_barrier) or np.isnan(ref_barrier)):
            barrier_errors.append(abs(pred_barrier - ref_barrier))

    energy_mae = float(np.mean(e_abs_errors)) if e_abs_errors else float("nan")
    force_mae  = float(np.mean(f_abs_errors)) if f_abs_errors else float("nan")
    barrier_mae = float(np.mean(barrier_errors)) if barrier_errors else float("nan")
    ts_force_mae = float(np.mean(ts_f_errors)) if ts_f_errors else float("nan")

    result = {
        "energy_mae": energy_mae,
        "force_mae":  force_mae,
        "barrier_mae": barrier_mae,
        "ts_force_mae": ts_force_mae,
        "n_frames": len(indices),
        "n_reactions": len(rxn_to_indices),
    }

    if timing and n_steps > 0:
        ms_per_atom_step = 1000.0 * total_time_s / max(total_atoms, 1)
        result["runtime_per_atom_step_ms"] = ms_per_atom_step

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
    p.add_argument("--ts-window",  type=float, default=0.1,
                   help="Fraction of IRC frames around midpoint for TS metric")
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

    # Load split IDs
    with open(splits_json) as f:
        splits = json.load(f)

    id_rxns  = set(splits.get("test_id_same_family", []))
    ood_rxns = set(splits.get("test_ood_family", []))
    val_rxns = set(splits.get("val_id", []))

    # Build datasets
    test_ds = Transition1xDataset(h5_path, splits=["test", "val", "data"])

    def idx_for(rxn_ids):
        return [i for i, entry in enumerate(test_ds._index) if entry[2] in rxn_ids]

    id_indices  = idx_for(id_rxns)
    ood_indices = idx_for(ood_rxns)
    val_indices = idx_for(val_rxns)

    print(f"[eval] {model_name}: ID={len(id_indices)} frames, "
          f"OOD={len(ood_indices)} frames, val={len(val_indices)} frames")

    # Evaluate
    id_metrics  = evaluate(model, atom_ref, test_ds, id_indices,  device,
                           ts_window=a.ts_window, timing=a.timing)
    ood_metrics = evaluate(model, atom_ref, test_ds, ood_indices, device,
                           ts_window=a.ts_window, timing=False)

    # OOD degradation: ratio of force MAE OOD vs ID
    ood_deg = float("nan")
    if id_metrics["force_mae"] > 0 and not np.isnan(ood_metrics["force_mae"]):
        ood_deg = ood_metrics["force_mae"] / id_metrics["force_mae"]

    row = {
        "model": model_name,
        "checkpoint": a.checkpoint,
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

    # Append to CSV
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
