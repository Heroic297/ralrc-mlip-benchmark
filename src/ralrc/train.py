"""Training loop for RALRC ablation models.

Loss:  L = w_E * MAE(E_pred, E_ref) + w_F * MAE(F_pred, F_ref)
  where w_F ~ 100 * w_E  (standard MLIP practice).

Split membership uses COMPOUND KEYS: "<hdf5_split>::<formula>::<rxn_id>"
so there are no false matches from bare rxn_id collisions across formulas
or HDF5 splits.

Batching strategy: molecules have variable atom counts so tensors cannot be
stacked. We use gradient accumulation over a list-collated mini-batch, which
keeps the model calls per-molecule while letting DataLoader workers prefetch
HDF5 frames asynchronously.

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
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

from .model_clean import ChargeAwarePotentialClean
from .transition1x import Transition1xDataset


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
HA_TO_EV = 27.2114


# ---------------------------------------------------------------------------
# Collation helpers
# ---------------------------------------------------------------------------

def _sample_to_tensors(sample: dict, device: torch.device):
    Z = torch.tensor(sample["z"],      dtype=torch.long,    device=device)
    R = torch.tensor(sample["pos"],    dtype=torch.float32, device=device)
    Q = torch.zeros((),                dtype=torch.long,    device=device)
    S = torch.ones((),                 dtype=torch.long,    device=device)
    # Transition1x stores energies in eV and forces in eV/Å — no unit conversion needed.
    # (HA_TO_EV was incorrectly applied here; removed in a prior commit.)
    E = torch.tensor(sample["energy"], dtype=torch.float32, device=device)
    F = torch.tensor(sample["forces"], dtype=torch.float32, device=device)
    return Z, R, Q, S, E, F


def _collate_variable_mols(batch):
    return batch


# ---------------------------------------------------------------------------
# Atomization reference energies
# ---------------------------------------------------------------------------

class AtomRefEnergy(nn.Module):
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
    E_pred, F_pred, E_ref, F_ref,
    w_E: float = 1.0, w_F: float = 100.0,
) -> torch.Tensor:
    return w_E * (E_pred - E_ref).abs() + w_F * (F_pred - F_ref).abs().mean()


def _compound_key(entry: tuple) -> str:
    return f"{entry[0]}::{entry[1]}::{entry[2]}"


def _fmt_seconds(s: float) -> str:
    """Format seconds as 'Xh Ym Zs' for display."""
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    elif m > 0:
        return f"{m}m {sec:02d}s"
    else:
        return f"{sec}s"


# ---------------------------------------------------------------------------
# Atom reference key parsing
# ---------------------------------------------------------------------------

# Accepts both symbol keys ("H", "C") and integer-string keys ("1", "6").
# e_ref.json from the fitted atom refs uses integer string keys; this mapping
# handles both so atom refs load correctly regardless of key format.
_SYMBOL_TO_Z: dict[str, int] = {
    "H": 1,  "1": 1,
    "C": 6,  "6": 6,
    "N": 7,  "7": 7,
    "O": 8,  "8": 8,
}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    h5_path: str,
    splits_json: str,
    model_cfg: dict,
    out_dir: str,
    atom_refs_json: str = None,
    seed: int = 17,
    epochs: int = 100,
    lr: float = 1e-3,
    w_E: float = 1.0,
    w_F: float = 100.0,
    batch_size: int = 32,
    num_workers: int = 0,
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

    with open(splits_json) as f:
        splits = json.load(f)
    train_keys = set(splits["train_id"])
    val_keys   = set(splits["val_id"])

    sample_train = next(iter(train_keys), None)
    if sample_train and "::" not in sample_train:
        raise ValueError(
            f"splits.json contains bare rxn_ids, not compound keys.\n"
            f"  Example: {sample_train!r}\n"
            f"  Expected: '<hdf5_split>::<formula>::<rxn_id>'\n"
            f"  Regenerate: python -m ralrc.split --h5 ... --out splits.json"
        )

    print("[train] Building index (all HDF5 splits)...")
    full_ds = Transition1xDataset(h5_path, splits=None)
    print(f"[train] Index size: {len(full_ds):,} frames")

    def filter_keys(ds, key_set):
        return [i for i, entry in enumerate(ds._index)
                if _compound_key(entry) in key_set]

    train_indices = filter_keys(full_ds, train_keys)
    val_indices   = filter_keys(full_ds, val_keys)

    if not train_indices:
        raise RuntimeError("[train] FATAL: 0 train frames matched. Stale splits.json?")
    if not val_indices:
        raise RuntimeError("[train] FATAL: 0 val frames matched.")

    if max_train_reactions is not None:
        seen: set = set()
        limited = []
        for i in train_indices:
            seen.add(_compound_key(full_ds._index[i]))
            limited.append(i)
            if len(seen) >= max_train_reactions:
                break
        train_indices = limited

    n_train_frames = len(train_indices)
    n_val_frames   = len(val_indices)
    n_batches_per_epoch = math.ceil(n_train_frames / batch_size)

    print(f"[train] Train: {n_train_frames:,} frames | "
          f"Val: {n_val_frames:,} frames | "
          f"Batches/epoch: {n_batches_per_epoch:,}")
    print(f"[train] batch_size={batch_size} | num_workers={num_workers} | "
          f"epochs={epochs} | device={device_str}")

    _use_workers = num_workers > 0
    train_loader = DataLoader(
        Subset(full_ds, train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_variable_mols,
        num_workers=num_workers,
        prefetch_factor=2 if _use_workers else None,
        persistent_workers=_use_workers,
        pin_memory=False,
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

    model = ChargeAwarePotentialClean(
        hidden=model_cfg.get("hidden", 64),
        use_coulomb=model_cfg.get("use_coulomb", True),
        charge_init_scale=model_cfg.get("charge_init_scale", None),
    ).to(device)
    atom_ref = AtomRefEnergy().to(device)

    # Load pre-fitted atom reference energies if provided.
    # Accepts both symbol keys ("H") and integer-string keys ("1") via
    # _SYMBOL_TO_Z, which maps both formats. Previously only symbol keys
    # were matched, causing e_ref.json (integer keys) to silently no-op.
    if atom_refs_json is not None and os.path.isfile(atom_refs_json):
        with open(atom_refs_json) as f:
            e_ref_data = json.load(f)
        loaded = {}
        with torch.no_grad():
            for sym, e_val in e_ref_data.items():
                z = _SYMBOL_TO_Z.get(sym)
                if z is not None:
                    atom_ref.ref.weight.data[z, 0] = float(e_val)
                    loaded[sym] = round(float(e_val), 2)
        print(f"[train] Loaded atom refs from {atom_refs_json}: {loaded}")
        if not loaded:
            print("[train] WARNING: No atom ref keys matched. E_MAE will be invalid.")

    # Only freeze the charge head and shield when BOTH use_charge=false AND
    # use_coulomb=false (i.e. pure local baseline with no charge machinery).
    # When use_charge=true but use_coulomb=false (charge_head_no_coulomb),
    # the charge head must remain trainable so charges are learned implicitly
    # through the energy head — otherwise this model is identical to
    # local_mace_style and the ablation is invalid.
    if not model_cfg.get("use_charge", True) and not model_cfg.get("use_coulomb", True):
        for p in model.charge_head.parameters():
            p.requires_grad_(False)
        model.shield.requires_grad_(False)

    all_params = list(model.parameters()) + list(atom_ref.parameters())
    optimizer = torch.optim.Adam(
        [p for p in all_params if p.requires_grad], lr=lr
    )

    best_val_force_mae = float("inf")
    best_epoch = -1

    epoch_times: deque = deque(maxlen=5)
    run_start = time.perf_counter()

    model_name = model_cfg.get("name", "model")
    print(f"\n{'='*60}")
    print(f"  {model_name}  |  seed={seed}  |  {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(epochs):
        cur_lr = cosine_lr(optimizer, epoch, epochs, lr)

        if lambda_coul_warmup_epochs > 0 and hasattr(model, "lambda_coul"):
            model.lambda_coul = float(min(epoch / lambda_coul_warmup_epochs, 1.0))

        model.train()
        atom_ref.train()
        train_loss_sum = 0.0
        n_train = 0
        epoch_start = time.perf_counter()

        if epoch_times:
            avg_epoch_s = sum(epoch_times) / len(epoch_times)
            epochs_left = epochs - epoch
            eta_run = _fmt_seconds(avg_epoch_s * epochs_left)
        else:
            eta_run = "estimating..."

        bar_desc = f"Ep {epoch+1:>3}/{epochs}"
        with tqdm(
            train_loader,
            desc=bar_desc,
            total=n_batches_per_epoch,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        ) as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                batch_loss = 0.0
                n_mol = len(batch)

                for sample in batch:
                    Z, R, Q, S, E_ref, F_ref = _sample_to_tensors(sample, device)
                    R.requires_grad_(True)
                    out = model.forward(Z, R, Q, S, compute_forces=True)
                    E_ref_corr = E_ref - atom_ref(Z)
                    loss = energy_force_loss(
                        out["energy"], out["forces"],
                        E_ref_corr, F_ref,
                        w_E=w_E, w_F=w_F,
                    ) / n_mol
                    loss.backward()
                    batch_loss += loss.item() * n_mol

                nn.utils.clip_grad_norm_(
                    [p for p in all_params if p.requires_grad and p.grad is not None],
                    max_norm=10.0,
                )
                optimizer.step()

                train_loss_sum += batch_loss
                n_train += n_mol

                pbar.set_postfix({
                    "loss": f"{train_loss_sum / max(n_train, 1):.4f}",
                    "lr":   f"{cur_lr:.1e}",
                    "run_eta": eta_run,
                }, refresh=False)

        epoch_wall = time.perf_counter() - epoch_start
        epoch_times.append(epoch_wall)

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

        avg_epoch_s = sum(epoch_times) / len(epoch_times)
        epochs_left = epochs - (epoch + 1)
        eta_run_str = _fmt_seconds(avg_epoch_s * epochs_left)
        elapsed_str = _fmt_seconds(time.perf_counter() - run_start)

        ckpt_flag = ""
        if val_f_mae < best_val_force_mae:
            best_val_force_mae = val_f_mae
            best_epoch = epoch
            ckpt_flag = "  [*] saved best"
            torch.save(
                {"model": model.state_dict(), "atom_ref": atom_ref.state_dict(),
                 "epoch": epoch, "val_force_mae": val_f_mae, "cfg": model_cfg},
                os.path.join(out_dir, "best.pt"),
            )

        print(
            f"  Ep {epoch+1:>3}/{epochs} "
            f"| loss={train_loss_sum/max(n_train,1):.4f} "
            f"| E_MAE={val_e_mae:.4f}eV "
            f"| F_MAE={val_f_mae:.4f}eV/A "
            f"| {_fmt_seconds(epoch_wall)}/ep "
            f"| elapsed={elapsed_str} "
            f"| ETA={eta_run_str}"
            f"{ckpt_flag}"
        )

        log_entry = {
            "epoch": epoch,
            "lr": cur_lr,
            "lambda_coul": getattr(model, "lambda_coul", 1.0),
            "train_loss": train_loss_sum / max(n_train, 1),
            "val_energy_mae_eV": val_e_mae,
            "val_force_mae_eV_ang": val_f_mae,
            "epoch_wall_s": epoch_wall,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    total_wall = time.perf_counter() - run_start
    print(f"\n[train] Done: {model_name} | "
          f"best F_MAE={best_val_force_mae:.4f} eV/A @ epoch {best_epoch} | "
          f"total time={_fmt_seconds(total_wall)}")

    summary = {
        "best_val_force_mae_eV_ang": best_val_force_mae,
        "best_epoch": best_epoch,
        "n_train_frames": n_train,
        "n_val_frames": n_val,
        "seed": seed,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "total_wall_s": total_wall,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train a RALRC ablation model.")
    p.add_argument("--config",      required=True)
    p.add_argument("--seed",        type=int, default=17)
    p.add_argument("--h5",          default=None)
    p.add_argument("--splits",      default=None)
    p.add_argument("--atom-refs",   default=None, dest="atom_refs")
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--device",      default=None)
    p.add_argument("--batch-size",  type=int, default=None, dest="batch_size")
    p.add_argument("--num-workers", type=int, default=None, dest="num_workers")
    a = p.parse_args()

    cfg         = yaml.safe_load(open(a.config))
    h5_path     = a.h5         or cfg.get("h5_path",    "data/transition1x.h5")
    splits_json = a.splits     or cfg.get("splits_json", "splits.json")
    atom_refs   = a.atom_refs  or cfg.get("atom_refs",   None)
    epochs      = a.epochs     or cfg.get("epochs",      100)
    device_str  = a.device     or cfg.get("device",      "cuda" if torch.cuda.is_available() else "cpu")
    batch_size  = a.batch_size or cfg.get("batch_size",  32)
    num_workers = a.num_workers if a.num_workers is not None else cfg.get("num_workers", 0)
    out_dir     = os.path.join("runs", cfg.get("name", "model"), f"seed{a.seed}")

    train(
        h5_path=h5_path,
        splits_json=splits_json,
        model_cfg=cfg,
        out_dir=out_dir,
        atom_refs_json=atom_refs,
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
