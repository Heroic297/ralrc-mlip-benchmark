"""Training loop for RALRC ablation models.

Loss:  L = w_E * MAE(E_pred, E_ref) + w_F * MAE(F_pred, F_ref)
  where w_F ~ 100 * w_E  (standard MLIP practice).

Split membership uses COMPOUND KEYS: "<hdf5_split>::<formula>::<rxn_id>"
so there are no false matches from bare rxn_id collisions across formulas
or HDF5 splits.

Batching strategy: molecules have variable atom counts so tensors cannot be
stacked. We use gradient accumulation over a list-collated mini-batch, which
keeps the model calls per-molecule while letting DataLoader workers prefetch
HDF5 frames asynchronously. This overlaps I/O with GPU compute and is the
primary speedup lever on a single GPU with fast storage.

Usage:
    python -m ralrc.train --config configs/learned_charge_coulomb.yaml --seed 17
"""
from __future__ import annotations

import argparse
import json
import os
import time
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from .model_clean import ChargeAwarePotentialClean
from .transition1x import Transition1xDataset


# ---------------------------------------------------------------------------
# Unit conversion: Transition1x stores energies/forces in Hartree(/Å)
# We train in eV (1 Ha = 27.2114 eV)
# ---------------------------------------------------------------------------
HA_TO_EV = 27.2114


# ---------------------------------------------------------------------------
# Collation helper
# ---------------------------------------------------------------------------

def _sample_to_tensors(sample: dict, device: torch.device):
    """Convert a raw Transition1x sample dict to torch tensors on device."""
    Z = torch.tensor(sample["z"], dtype=torch.long, device=device)
    R = torch.tensor(sample["pos"], dtype=torch.float32, device=device)
    Q = torch.zeros((), dtype=torch.long, device=device)
    S = torch.ones((), dtype=torch.long, device=device)
    E = torch.tensor(sample["energy"] * HA_TO_EV, dtype=torch.float32, device=device)
    F = torch.tensor(sample["forces"] * HA_TO_EV, dtype=torch.float32, device=device)
    return Z, R, Q, S, E, F


def _collate_variable_mols(batch):
    """Return list of sample dicts unchanged.

    Molecules have variable atom counts so stacking is not possible without
    padding. Gradient accumulation over this list achieves the same effect as
    a true batch while keeping per-molecule model calls intact.
    """
    return batch


# ---------------------------------------------------------------------------
# Atomization reference energies
# ---------------------------------------------------------------------------

class AtomRefEnergy(nn.Module):
    """Learnable per-element reference energies."""
    def __init__(self, n_elements: int = 119):
        super().__init__()
        self.ref = nn.Embedding(n_elements, 1)
        nn.init.zeros_(self.ref.weight)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.ref(Z).sum()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def cosine_lr(optimizer, epoch: int, total_epochs: int, lr_max: float, lr_min: float = 1e-6):
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / total_epochs))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def energy_force_loss(
    E_pred: torch.Tensor,
    F_pred: torch.Tensor,
    E_ref: torch.Tensor,
    F_ref: torch.Tensor,
    w_E: float = 1.0,
    w_F: float = 100.0,
) -> torch.Tensor:
    loss_E = (E_pred - E_ref).abs()
    loss_F = (F_pred - F_ref).abs().mean()
    return w_E * loss_E + w_F * loss_F


# ---------------------------------------------------------------------------
# Compound key helper
# ---------------------------------------------------------------------------

def _compound_key(entry: tuple) -> str:
    """Build a globally unique key from a dataset index entry.

    entry = (split, formula, rxn_id, frame_idx, endpoint)
    key   = "<split>::<formula>::<rxn_id>"
    """
    return f"{entry[0]}::{entry[1]}::{entry[2]}"


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    h5_path: str,
    splits_json: str,
    model_cfg: dict,
    out_dir: str,
    seed: int = 17,
    epochs: int = 100,
    lr: float = 1e-3,
    w_E: float = 1.0,
    w_F: float = 100.0,
    batch_size: int = 32,
    num_workers: int = 4,
    max_train_reactions: Optional[int] = None,
    lambda_coul_warmup_epochs: int = 0,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device_str)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.jsonl")

    # Load split compound keys
    with open(splits_json) as f:
        splits = json.load(f)
    train_keys = set(splits["train_id"])
    val_keys   = set(splits["val_id"])

    # Sanity check: splits must use compound keys
    def _looks_like_compound_key(s: str) -> bool:
        return "::" in s

    sample_train = next(iter(train_keys), None)
    if sample_train is not None and not _looks_like_compound_key(sample_train):
        raise ValueError(
            f"splits.json appears to contain bare rxn_ids, not compound keys.\n"
            f"  Example entry: {sample_train!r}\n"
            f"  Expected format: '<hdf5_split>::<formula>::<rxn_id>'\n"
            f"  Regenerate splits with: python -m ralrc.split --h5 ... --out splits.json"
        )

    # Build dataset over ALL HDF5 splits so every compound key is visible
    print("[train] Building index (all HDF5 splits)...")
    full_ds = Transition1xDataset(h5_path, splits=None)
    print(f"[train] Index size: {len(full_ds)} frames")

    def filter_by_compound_key(ds, key_set):
        return [i for i, entry in enumerate(ds._index)
                if _compound_key(entry) in key_set]

    train_indices = filter_by_compound_key(full_ds, train_keys)
    val_indices   = filter_by_compound_key(full_ds, val_keys)

    if len(train_indices) == 0:
        raise RuntimeError(
            f"[train] FATAL: 0 train frames matched. "
            f"Check that splits.json was generated with the current HDF5."
        )
    if len(val_indices) == 0:
        raise RuntimeError(
            f"[train] FATAL: 0 val frames matched. Same issue."
        )

    if max_train_reactions is not None:
        seen_rxns: set = set()
        limited = []
        for i in train_indices:
            rxn_key = _compound_key(full_ds._index[i])
            seen_rxns.add(rxn_key)
            limited.append(i)
            if len(seen_rxns) >= max_train_reactions:
                break
        train_indices = limited

    print(f"[train] Train frames: {len(train_indices)}, Val frames: {len(val_indices)}")
    print(f"[train] DataLoader: batch_size={batch_size}, num_workers={num_workers}")

    # Build DataLoaders once — shuffle=True reshuffles automatically each epoch.
    # list collation handles variable atom counts without padding.
    _use_workers = num_workers > 0
    train_loader = DataLoader(
        Subset(full_ds, train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_variable_mols,
        num_workers=num_workers,
        prefetch_factor=2 if _use_workers else None,
        persistent_workers=_use_workers,
        pin_memory=False,  # list-of-dicts cannot be pinned
    )
    val_loader = DataLoader(
        Subset(full_ds, val_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_variable_mols,
        num_workers=num_workers,
        prefetch_factor=2 if _use_workers else None,
        persistent_workers=_use_workers,
        pin_memory=False,
    )

    # Model
    charge_init_scale = model_cfg.get("charge_init_scale", None)
    model = ChargeAwarePotentialClean(
        hidden=model_cfg.get("hidden", 64),
        use_coulomb=model_cfg.get("use_coulomb", True),
        charge_init_scale=charge_init_scale,
    ).to(device)
    atom_ref = AtomRefEnergy().to(device)

    if not model_cfg.get("use_charge", True):
        for p in model.charge_head.parameters():
            p.requires_grad_(False)
        model.shield.requires_grad_(False)

    params = list(model.parameters()) + list(atom_ref.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    best_val_force_mae = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        cur_lr = cosine_lr(optimizer, epoch, epochs, lr)

        # Lambda_coul warmup: ramp Coulomb scale from 0 → 1 over warmup_epochs
        if lambda_coul_warmup_epochs > 0 and hasattr(model, "lambda_coul"):
            model.lambda_coul = float(min(epoch / lambda_coul_warmup_epochs, 1.0))

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        model.train()
        atom_ref.train()
        train_loss_sum = 0.0
        n_train = 0

        for batch in train_loader:
            # Gradient accumulation over the variable-size mini-batch.
            # Grads are accumulated across all molecules before the optimizer
            # step, matching the semantics of a true batched forward pass.
            optimizer.zero_grad()
            batch_loss = 0.0
            n_mol = len(batch)

            for sample in batch:
                Z, R, Q, S, E_ref, F_ref = _sample_to_tensors(sample, device)
                R.requires_grad_(True)
                out = model.forward(Z, R, Q, S, compute_forces=True)
                E_ref_corr = E_ref - atom_ref(Z)

                # Normalize loss by batch size so effective LR is independent
                # of batch_size (equivalent to mean reduction across the batch).
                loss = energy_force_loss(
                    out["energy"], out["forces"],
                    E_ref_corr, F_ref,
                    w_E=w_E, w_F=w_F,
                ) / n_mol
                loss.backward()
                batch_loss += loss.item() * n_mol  # track un-normalized for logging

            nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

            train_loss_sum += batch_loss
            n_train += n_mol

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        atom_ref.eval()
        val_e_mae = 0.0
        val_f_mae = 0.0
        n_val = 0

        for batch in val_loader:
            for sample in batch:
                Z, R, Q, S, E_ref, F_ref = _sample_to_tensors(sample, device)
                with torch.enable_grad():
                    out = model.forward(Z, R, Q, S, compute_forces=True)
                E_ref_corr = E_ref - atom_ref(Z).detach()
                val_e_mae += (out["energy"].detach() - E_ref_corr).abs().item()
                val_f_mae += (out["forces"].detach() - F_ref).abs().mean().item()
                n_val += 1

        val_e_mae /= max(n_val, 1)
        val_f_mae /= max(n_val, 1)

        log_entry = {
            "epoch": epoch,
            "lr": cur_lr,
            "lambda_coul": getattr(model, "lambda_coul", 1.0),
            "train_loss": train_loss_sum / max(n_train, 1),
            "val_energy_mae_eV": val_e_mae,
            "val_force_mae_eV_ang": val_f_mae,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:4d}/{epochs} | lr={cur_lr:.2e} | "
                  f"train_loss={log_entry['train_loss']:.4f} | "
                  f"val_E_MAE={val_e_mae:.4f} eV | val_F_MAE={val_f_mae:.4f} eV/Å")

        if val_f_mae < best_val_force_mae:
            best_val_force_mae = val_f_mae
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "atom_ref": atom_ref.state_dict(),
                 "epoch": epoch, "val_force_mae": val_f_mae, "cfg": model_cfg},
                os.path.join(out_dir, "best.pt"),
            )

    print(f"[train] Done. Best val force MAE = {best_val_force_mae:.4f} eV/Å at epoch {best_epoch}")
    summary = {
        "best_val_force_mae_eV_ang": best_val_force_mae,
        "best_epoch": best_epoch,
        "n_train_frames": n_train,
        "n_val_frames": n_val,
        "seed": seed,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train a RALRC ablation model.")
    p.add_argument("--config",       required=True)
    p.add_argument("--seed",         type=int, default=17)
    p.add_argument("--h5",           default=None)
    p.add_argument("--splits",       default=None)
    p.add_argument("--epochs",       type=int, default=None)
    p.add_argument("--device",       default=None)
    p.add_argument("--batch-size",   type=int, default=None, dest="batch_size")
    p.add_argument("--num-workers",  type=int, default=None, dest="num_workers")
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    h5_path     = a.h5          or cfg.get("h5_path",    "data/transition1x.h5")
    splits_json = a.splits      or cfg.get("splits_json", "splits.json")
    epochs      = a.epochs      or cfg.get("epochs",      100)
    device_str  = a.device      or cfg.get("device",      "cuda" if torch.cuda.is_available() else "cpu")
    batch_size  = a.batch_size  or cfg.get("batch_size",  32)
    num_workers = a.num_workers if a.num_workers is not None else cfg.get("num_workers", 4)
    out_dir = os.path.join("runs", cfg.get("name", "model"), f"seed{a.seed}")

    print(f"\n=== Training {cfg.get('name','model')} | seed={a.seed} | device={device_str} ===")
    train(
        h5_path=h5_path,
        splits_json=splits_json,
        model_cfg=cfg,
        out_dir=out_dir,
        seed=a.seed,
        epochs=epochs,
        lr=cfg.get("lr", 1e-3),
        w_E=cfg.get("w_E", 1.0),
        w_F=cfg.get("w_F", 100.0),
        batch_size=batch_size,
        num_workers=num_workers,
        max_train_reactions=cfg.get("max_train_reactions", None),
        lambda_coul_warmup_epochs=cfg.get("lambda_coul_warmup_epochs", 0),
        device_str=device_str,
    )


if __name__ == "__main__":
    main()
