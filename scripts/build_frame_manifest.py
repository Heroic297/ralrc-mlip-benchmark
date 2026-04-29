import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

ENERGY_KEY = "wB97x_6-31G(d).energy"
FORCES_KEY = "wB97x_6-31G(d).forces"

def choose_indices(n_frames, stride, include_ends=True):
    idx = list(range(0, n_frames, stride))
    if include_ends:
        idx.extend([0, n_frames - 1])
    return sorted(set(i for i in idx if 0 <= i < n_frames))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max-reactions-per-split", type=int, default=None)
    parser.add_argument("--out", default="reports/dataset_checks/frame_manifest.csv")
    args = parser.parse_args()

    rows = []

    with h5py.File(args.h5, "r") as f:
        for split in args.splits:
            if split not in f:
                continue

            n_seen = 0
            split_grp = f[split]

            for formula in tqdm(list(split_grp.keys()), desc=f"split={split}"):
                formula_grp = split_grp[formula]

                for rxn_id, rxn_grp in formula_grp.items():
                    if args.max_reactions_per_split is not None and n_seen >= args.max_reactions_per_split:
                        break

                    if not all(k in rxn_grp for k in ["atomic_numbers", "positions", ENERGY_KEY, FORCES_KEY]):
                        continue

                    z = rxn_grp["atomic_numbers"][()]
                    n_atoms = int(len(z))
                    n_frames = int(rxn_grp["positions"].shape[0])
                    energies = rxn_grp[ENERGY_KEY][()]
                    forces = rxn_grp[FORCES_KEY]

                    e0 = float(energies[0])
                    e_max = float(np.max(energies))
                    traj_barrier_raw = e_max - e0

                    ts_e_raw = None
                    reactant_e_raw = None
                    endpoint_barrier_raw = None

                    if "transition_state" in rxn_grp and ENERGY_KEY in rxn_grp["transition_state"]:
                        ts_e_raw = float(rxn_grp["transition_state"][ENERGY_KEY][0])

                    if "reactant" in rxn_grp and ENERGY_KEY in rxn_grp["reactant"]:
                        reactant_e_raw = float(rxn_grp["reactant"][ENERGY_KEY][0])

                    if ts_e_raw is not None and reactant_e_raw is not None:
                        endpoint_barrier_raw = ts_e_raw - reactant_e_raw

                    for i in choose_indices(n_frames, args.stride):
                        f_i = forces[i]
                        force_l2 = float(np.linalg.norm(f_i.reshape(-1)))
                        force_rms = float(np.sqrt(np.mean(f_i ** 2)))
                        xi_index = i / max(n_frames - 1, 1)

                        rows.append({
                            "split": split,
                            "formula": formula,
                            "rxn_id": rxn_id,
                            "frame_idx": i,
                            "n_frames": n_frames,
                            "n_atoms": n_atoms,
                            "xi_index": xi_index,
                            "energy_raw": float(energies[i]),
                            "energy_rel_first_raw": float(energies[i] - e0),
                            "traj_barrier_raw": traj_barrier_raw,
                            "endpoint_barrier_raw": endpoint_barrier_raw,
                            "force_l2": force_l2,
                            "force_rms": force_rms,
                        })

                    n_seen += 1

                if args.max_reactions_per_split is not None and n_seen >= args.max_reactions_per_split:
                    break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df):,} rows to {out}")
    print(df.groupby("split").size())
    print(df.describe(include="all").T)

if __name__ == "__main__":
    main()
