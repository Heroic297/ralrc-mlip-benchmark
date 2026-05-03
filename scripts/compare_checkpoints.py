"""
compare_checkpoints.py

Compares parameter norms between local_mace_style and charge_head_no_coulomb
best.pt checkpoints to determine whether the charge head is active/dead
in the no-coulomb ablation (which tied the local baseline at exactly 0.1633).

Usage:
    python scripts/compare_checkpoints.py

Outputs:
    - Per-layer norm differences (only layers with >1% relative difference)
    - Charge head total norm for each checkpoint
    - Verdict: whether charge_head_no_coulomb is a valid distinct ablation
"""
import torch
from pathlib import Path

CHECKPOINTS = {
    "local_mace_style":       "runs/local_mace_style/seed17/best.pt",
    "charge_head_no_coulomb": "runs/charge_head_no_coulomb/seed17/best.pt",
    "learned_charge_coulomb": "runs/learned_charge_coulomb/seed17/best.pt",
}


def load_model_state(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt["model"]


def param_norms(state: dict) -> dict[str, float]:
    return {k: v.norm().item() for k, v in state.items()}


def charge_head_total_norm(state: dict) -> float:
    return sum(
        v.norm().item() ** 2
        for k, v in state.items()
        if "charge_head" in k
    ) ** 0.5


def main():
    states = {}
    for name, path in CHECKPOINTS.items():
        p = Path(path)
        if not p.exists():
            print(f"[SKIP] {name}: {path} not found")
            continue
        states[name] = load_model_state(path)
        print(f"[OK]   {name}: loaded from {path}")

    print()

    # Charge head norms
    print("=== Charge Head Total Norm ===")
    for name, state in states.items():
        norm = charge_head_total_norm(state)
        print(f"  {name:35s}: {norm:.6f}")

    print()

    # Pairwise layer norm diff: local vs no_coulomb
    a_name, b_name = "local_mace_style", "charge_head_no_coulomb"
    if a_name in states and b_name in states:
        print(f"=== Layer Norm Diff: {a_name} vs {b_name} ===")
        a_norms = param_norms(states[a_name])
        b_norms = param_norms(states[b_name])
        diffs_found = False
        for key in a_norms:
            if key not in b_norms:
                continue
            a_val, b_val = a_norms[key], b_norms[key]
            ref = max(abs(a_val), abs(b_val), 1e-8)
            rel_diff = abs(a_val - b_val) / ref
            if rel_diff > 0.01:  # >1% relative difference
                print(f"  {key:50s}  local={a_val:.4f}  no_coul={b_val:.4f}  rel_diff={rel_diff:.1%}")
                diffs_found = True
        if not diffs_found:
            print("  No layers differ by >1%. Models are effectively identical.")
            print("  VERDICT: charge_head_no_coulomb ablation is INVALID (charge head is dead/unused).")
        else:
            print("  VERDICT: Models differ — charge head is active in charge_head_no_coulomb.")
            print("  The 0.1633 tie is a genuine result: learned charges without Coulomb add no benefit.")

    print()

    # Also compare learned_charge_coulomb charge head norm vs the others
    if "learned_charge_coulomb" in states:
        print("=== Learned Charge Coulomb vs No-Coulomb (charge head only) ===")
        for other in ["local_mace_style", "charge_head_no_coulomb"]:
            if other not in states:
                continue
            lcc_norm = charge_head_total_norm(states["learned_charge_coulomb"])
            oth_norm = charge_head_total_norm(states[other])
            print(f"  learned_charge_coulomb charge_head norm : {lcc_norm:.6f}")
            print(f"  {other:35s} charge_head norm : {oth_norm:.6f}")
            ratio = lcc_norm / max(oth_norm, 1e-8)
            print(f"  Ratio (lcc / other)                     : {ratio:.2f}x")
            break


if __name__ == "__main__":
    main()
