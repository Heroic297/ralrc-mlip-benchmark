"""Dataset summary/check script for Transition1x.h5.

Reports:
  - Total formulas, reactions, frames per split
  - Atom-count distribution (min/max/mean/p50/p95)
  - Frame-count distribution per reaction
  - Reactions with missing required keys
  - Endpoint coverage (reactant / product / transition_state)
  - Energy and force magnitude statistics at TS vs non-TS frames

Usage:
  python scripts/dataset_summary.py data/raw/Transition1x.h5
  python scripts/dataset_summary.py data/raw/Transition1x.h5 --out reports/dataset_checks/summary.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ENERGY_KEY = "wB97x_6-31G(d).energy"
FORCES_KEY = "wB97x_6-31G(d).forces"
REQUIRED = {"atomic_numbers", "positions", ENERGY_KEY, FORCES_KEY}
ENDPOINTS = ("reactant", "product", "transition_state")


def percentile_summary(arr: list | np.ndarray) -> dict:
    a = np.array(arr, dtype=float)
    return {
        "min": float(np.min(a)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", help="Path to Transition1x.h5")
    parser.add_argument("--out", default=None, help="Write JSON report here")
    parser.add_argument(
        "--max-reactions", type=int, default=None,
        help="Limit reactions per split for fast smoke-check"
    )
    args = parser.parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        sys.exit(f"ERROR: file not found: {h5_path}")

    report = {}

    with h5py.File(h5_path, "r") as f:
        report["root_keys"] = list(f.keys())

        for split in f.keys():
            split_grp = f[split]
            print(f"\n=== split: {split} ===")

            formulas = list(split_grp.keys())
            n_formulas = len(formulas)

            atom_counts = []
            frame_counts = []
            n_rxns = 0
            n_frames_total = 0
            missing_rxns = []

            ep_counts = defaultdict(int)   # reactant / product / transition_state

            # TS energy barrier and force-norm accumulators
            ts_force_norms = []
            nts_force_norms = []
            energy_barriers = []  # E_TS - E_reactant

            rxn_iter = 0
            for formula in formulas:
                frml_grp = split_grp[formula]
                if not isinstance(frml_grp, h5py.Group):
                    continue
                for rxn_id, rxn_grp in frml_grp.items():
                    if not isinstance(rxn_grp, h5py.Group):
                        continue

                    if args.max_reactions and rxn_iter >= args.max_reactions:
                        break
                    rxn_iter += 1

                    missing = [k for k in REQUIRED if k not in rxn_grp]
                    if missing:
                        missing_rxns.append({"split": split, "formula": formula,
                                              "rxn_id": rxn_id, "missing": missing})
                        continue

                    n_atoms = rxn_grp["atomic_numbers"].shape[0]
                    n_frames = rxn_grp["positions"].shape[0]
                    atom_counts.append(n_atoms)
                    frame_counts.append(n_frames)
                    n_rxns += 1
                    n_frames_total += n_frames

                    # force norms for non-TS trajectory frames
                    forces = rxn_grp[FORCES_KEY][()]  # (n_frames, n_atoms, 3)
                    fnorms = np.linalg.norm(forces.reshape(n_frames, -1), axis=1)  # (n_frames,)
                    nts_force_norms.extend(fnorms.tolist())

                    # endpoints
                    for ep in ENDPOINTS:
                        if ep in rxn_grp:
                            ep_counts[ep] += 1

                    # TS stats
                    if "transition_state" in rxn_grp:
                        ts_grp = rxn_grp["transition_state"]
                        if FORCES_KEY in ts_grp and ENERGY_KEY in ts_grp:
                            ts_f = ts_grp[FORCES_KEY][()]   # (1, n_atoms, 3)
                            ts_fn = float(np.linalg.norm(ts_f.reshape(-1)))
                            ts_force_norms.append(ts_fn)

                        if ENERGY_KEY in ts_grp and "reactant" in rxn_grp:
                            r_grp = rxn_grp["reactant"]
                            if ENERGY_KEY in r_grp:
                                e_ts = float(ts_grp[ENERGY_KEY][0])
                                e_r = float(r_grp[ENERGY_KEY][0])
                                energy_barriers.append(e_ts - e_r)

            split_report = {
                "n_formulas": n_formulas,
                "n_reactions": n_rxns,
                "n_frames_total": n_frames_total,
                "missing_key_reactions": len(missing_rxns),
                "atom_count_stats": percentile_summary(atom_counts) if atom_counts else {},
                "frame_count_stats": percentile_summary(frame_counts) if frame_counts else {},
                "endpoint_coverage": dict(ep_counts),
            }

            if ts_force_norms:
                split_report["ts_force_norm_stats"] = percentile_summary(ts_force_norms)
            if nts_force_norms:
                split_report["nts_force_norm_stats"] = percentile_summary(nts_force_norms[:100000])
            if energy_barriers:
                split_report["energy_barrier_stats_hartree"] = percentile_summary(energy_barriers)

            report[split] = split_report

            # Print to stdout
            print(f"  formulas:       {n_formulas}")
            print(f"  reactions:      {n_rxns}")
            print(f"  total frames:   {n_frames_total}")
            print(f"  missing-key rxn:{len(missing_rxns)}")
            if atom_counts:
                s = percentile_summary(atom_counts)
                print(f"  atom count:     min={s['min']:.0f} p50={s['p50']:.0f} "
                      f"p95={s['p95']:.0f} max={s['max']:.0f}")
            if frame_counts:
                s = percentile_summary(frame_counts)
                print(f"  frames/rxn:     min={s['min']:.0f} p50={s['p50']:.0f} "
                      f"p95={s['p95']:.0f} max={s['max']:.0f}")
            print(f"  endpoint coverage: " + ", ".join(f"{k}={v}" for k, v in ep_counts.items()))
            if energy_barriers:
                s = percentile_summary(energy_barriers)
                print(f"  barrier (Ha):   min={s['min']:.4f} p50={s['p50']:.4f} max={s['max']:.4f}")

        if missing_rxns:
            report["missing_key_detail"] = missing_rxns[:50]  # first 50 only

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fout:
            json.dump(report, fout, indent=2)
        print(f"\nReport written to {out_path}")

    return report


if __name__ == "__main__":
    main()
