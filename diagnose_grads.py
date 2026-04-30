"""diagnose_grads.py — one-shot gradient health check for RALRC training.

Loads a checkpoint (or fresh init), runs 3 training batches, and reports:
  1. Whether out['forces'] has a grad_fn during training (must be non-None)
  2. Per-layer gradient magnitudes after force-only and energy-only backward
  3. Dataset statistics: mean |E|, mean |F| in raw units

Usage:
    python diagnose_grads.py \\
        --config configs/local_mace_style.yaml \\
        --h5 data/transition1x.h5 \\
        --splits splits_pilot.json \\
        --atom-refs runs/e_ref.json \\
        --checkpoint runs/local_mace_style/seed17/best.pt
"""
import argparse
import json
import sys
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# ---- resolve src layout ----
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ralrc.model_clean import ChargeAwarePotentialClean
from ralrc.transition1x import Transition1xDataset
from ralrc.train import AtomRefEnergy, _collate_variable_mols, _compound_key


def _load_checkpoint(model, atom_ref, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    if "atom_ref" in ckpt:
        atom_ref.load_state_dict(ckpt["atom_ref"], strict=False)
    print(f"[diag] Loaded checkpoint: {ckpt_path}")


def _load_atom_refs(atom_ref, refs_json, device):
    with open(refs_json) as f:
        data = json.load(f)
    _sym2z = {"H": 1, "C": 6, "N": 7, "O": 8}
    with torch.no_grad():
        for sym, val in data.items():
            z = _sym2z.get(sym)
            if z is not None:
                atom_ref.ref.weight[z] = torch.tensor(float(val))
    print(f"[diag] Loaded atom refs: { {s: round(v,2) for s,v in data.items()} }")


def _sample_to_tensors(sample, device):
    Z = torch.tensor(sample["z"],      dtype=torch.long,    device=device)
    R = torch.tensor(sample["pos"],    dtype=torch.float32, device=device)
    Q = torch.zeros((),                dtype=torch.long,    device=device)
    S = torch.ones((),                 dtype=torch.long,    device=device)
    # No HA_TO_EV — Transition1x is eV / eV/Å
    E = torch.tensor(sample["energy"], dtype=torch.float32, device=device)
    F = torch.tensor(sample["forces"], dtype=torch.float32, device=device)
    return Z, R, Q, S, E, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True)
    ap.add_argument("--h5",         required=True)
    ap.add_argument("--splits",     required=True)
    ap.add_argument("--atom-refs",  default=None, dest="atom_refs")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-batches",  type=int, default=3, dest="n_batches")
    ap.add_argument("--batch-size", type=int, default=4,  dest="batch_size")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))

    model = ChargeAwarePotentialClean(
        hidden=cfg.get("hidden", 64),
        use_coulomb=cfg.get("use_coulomb", True),
        charge_init_scale=cfg.get("charge_init_scale", None),
    ).to(device)
    atom_ref = AtomRefEnergy().to(device)

    if args.atom_refs:
        _load_atom_refs(atom_ref, args.atom_refs, device)
    if args.checkpoint:
        _load_checkpoint(model, atom_ref, args.checkpoint, device)

    with open(args.splits) as f:
        splits = json.load(f)
    val_keys = set(splits["val_id"])

    full_ds = Transition1xDataset(args.h5, splits=None)
    val_idx = [i for i, e in enumerate(full_ds._index) if _compound_key(e) in val_keys][:200]
    loader = DataLoader(
        Subset(full_ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_variable_mols,
        num_workers=0,
    )

    params = list(model.parameters()) + list(atom_ref.parameters())

    # --- dataset stats ---
    print("\n[diag] === Dataset statistics (first 200 val frames) ===")
    e_vals, f_vals = [], []
    for batch in loader:
        for s in batch:
            _, _, _, _, E, F = _sample_to_tensors(s, device)
            e_vals.append(E.item())
            f_vals.append(F.abs().mean().item())
        if len(e_vals) >= 200:
            break
    import statistics
    print(f"  E  : mean={statistics.mean(e_vals):.3f} eV  std={statistics.stdev(e_vals):.3f} eV")
    print(f"  |F|: mean={statistics.mean(f_vals):.4f} eV/Å  std={statistics.stdev(f_vals):.4f} eV/Å")

    # --- gradient check ---
    print(f"\n[diag] === Gradient check ({args.n_batches} batches, bs={args.batch_size}) ===")
    model.train()
    atom_ref.train()

    for b_idx, batch in enumerate(loader):
        if b_idx >= args.n_batches:
            break

        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        e_loss_total = torch.tensor(0.0, device=device)
        f_loss_total = torch.tensor(0.0, device=device)
        n_mol = len(batch)

        forces_grad_fn_present = []
        for sample in batch:
            Z, R, Q, S, E_ref, F_ref = _sample_to_tensors(sample, device)
            R.requires_grad_(True)
            out = model.forward(Z, R, Q, S, compute_forces=True)

            has_grad_fn = out["forces"].grad_fn is not None
            forces_grad_fn_present.append(has_grad_fn)

            E_ref_corr = E_ref - atom_ref(Z)
            e_loss = (out["energy"] - E_ref_corr).abs() / n_mol
            f_loss = (out["forces"] - F_ref).abs().mean() / n_mol
            e_loss_total = e_loss_total + e_loss
            f_loss_total = f_loss_total + f_loss

        all_have_grad_fn = all(forces_grad_fn_present)
        print(f"\n  Batch {b_idx}:")
        print(f"    forces.grad_fn present: {all_have_grad_fn}  "
              f"({'OK — force loss will backprop' if all_have_grad_fn else 'BUG — forces are detached, force loss is dead'})")
        print(f"    E_loss={e_loss_total.item():.6f}  F_loss={f_loss_total.item():.6f}  "
              f"ratio={f_loss_total.item()/max(e_loss_total.item(),1e-12):.1f}×")

        # Backward on force loss only
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        f_loss_total.backward(retain_graph=True)
        f_grads = {n: p.grad.abs().mean().item() if p.grad is not None else 0.0
                   for n, p in model.named_parameters()}

        # Backward on energy loss only
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        e_loss_total.backward()
        e_grads = {n: p.grad.abs().mean().item() if p.grad is not None else 0.0
                   for n, p in model.named_parameters()}

        print(f"    {'Layer':<45} {'F-grad':>10} {'E-grad':>10} {'ratio':>8}")
        print(f"    {'-'*45} {'-'*10} {'-'*10} {'-'*8}")
        for name in f_grads:
            fg = f_grads[name]
            eg = e_grads[name]
            ratio = fg / max(eg, 1e-20)
            flag = "  ← DEAD" if fg < 1e-10 else ""
            print(f"    {name:<45} {fg:>10.3e} {eg:>10.3e} {ratio:>8.2f}x{flag}")

    print("\n[diag] Done.")
    print("[diag] If F-grad is DEAD on all energy_head/message_pass layers → force loss not backpropping.")
    print("[diag] If E-grad >> F-grad everywhere → w_F=100 is not actually driving force training.")


if __name__ == "__main__":
    main()
