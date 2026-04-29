"""Diagnose frame and force balance in RALRC splits.

Understands COMPOUND KEYS ("<hdf5_split>::<formula>::<rxn_id>").

Usage:
    python scripts/diagnose_frames.py \\
        --h5 data/transition1x.h5 \\
        --splits splits.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np


def _compound_key(split: str, formula: str, rxn_id: str) -> str:
    return f"{split}::{formula}::{rxn_id}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",     required=True)
    p.add_argument("--splits", required=True)
    a = p.parse_args()

    with open(a.splits) as f:
        splits = json.load(f)

    id_keys  = set(splits.get("test_id_same_family", []))
    ood_keys = set(splits.get("test_ood_family", []))
    train_keys = set(splits.get("train_id", []))
    val_keys   = set(splits.get("val_id", []))

    all_key_sets = {
        "train": train_keys,
        "val":   val_keys,
        "test_id": id_keys,
        "test_ood": ood_keys,
    }

    # Check compound key format
    sample = next(iter(train_keys), None)
    if sample and "::" not in sample:
        print(f"[diagnose] WARNING: splits appear to use bare rxn_ids, not compound keys.")
        print(f"  Example: {sample!r}")
        print(f"  Regenerate splits with: python -m ralrc.split --h5 {a.h5} --out splits.json")
        return

    print(f"[diagnose] Split sizes (reactions):")
    for name, ks in all_key_sets.items():
        print(f"  {name}: {len(ks)} reactions")

    # Walk HDF5 and count matched frames per split
    frame_counts  = defaultdict(int)
    rxn_matches   = defaultdict(int)
    force_norms   = defaultdict(list)  # per split: list of per-frame mean |F|
    n_total = 0

    with h5py.File(a.h5, "r") as f:
        for hdf_split in f.keys():
            grp = f[hdf_split]
            if not hasattr(grp, 'items'):
                continue
            for formula, frml_grp in grp.items():
                if not hasattr(frml_grp, 'items'):
                    continue
                for rxn_id, rxn_grp in frml_grp.items():
                    if not hasattr(rxn_grp, 'items'):
                        continue
                    if "positions" not in rxn_grp:
                        continue
                    ck = _compound_key(hdf_split, formula, rxn_id)
                    n_frames = rxn_grp["positions"].shape[0]
                    n_total += n_frames

                    for split_name, key_set in all_key_sets.items():
                        if ck in key_set:
                            frame_counts[split_name] += n_frames
                            rxn_matches[split_name]  += 1
                            # Sample force norms for first 5 frames to keep it fast
                            fkey = "wB97x_6-31G(d).forces"
                            if fkey in rxn_grp:
                                n_sample = min(5, n_frames)
                                for fi in range(n_sample):
                                    fvec = rxn_grp[fkey][fi]  # (n_atoms, 3)
                                    mean_norm = float(np.linalg.norm(fvec, axis=1).mean())
                                    force_norms[split_name].append(mean_norm)
                            break  # a key can only be in one split

    print(f"\n[diagnose] Dataset total frames: {n_total}")
    print(f"[diagnose] Frame matches per split:")
    for name in ["train", "val", "test_id", "test_ood"]:
        fc = frame_counts.get(name, 0)
        rc = rxn_matches.get(name, 0)
        fn = force_norms.get(name, [])
        fn_mean = float(np.mean(fn)) if fn else float("nan")
        print(f"  {name:10s}: {fc:10d} frames | {rc:6d} reactions | "
              f"mean |F| ~ {fn_mean:.4f} Ha/Å (sampled)")

    # Sanity check
    any_zero = any(frame_counts.get(n, 0) == 0 for n in ["train", "val", "test_id", "test_ood"])
    if any_zero:
        print("\n[diagnose] FAIL: one or more splits have 0 matched frames.")
        print("  Possible causes:")
        print("  1. splits.json was generated from a different HDF5 file")
        print("  2. splits.json uses bare rxn_ids (no '::') instead of compound keys")
        print("  3. --max-reactions too small for OOD families")
    else:
        print("\n[diagnose] PASS: all splits have nonzero frames.")


if __name__ == "__main__":
    main()
