"""Diagnose frame count and force RMS imbalance across the pilot dataset.

Usage:
    python scripts/diagnose_frames.py --h5 data/transition1x.h5 --splits splits.json

Outputs:
    - Per-split frame count and median IRC path length
    - Per-split force RMS distribution (median and IQR)
    - Per-reaction frame count histogram
    - Flags if train has < 80% of val frame count (potential bug)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",     required=True)
    p.add_argument("--splits", required=True)
    a = p.parse_args()

    try:
        import h5py
    except ImportError:
        print("h5py required: pip install h5py")
        sys.exit(1)

    with open(a.splits) as f:
        splits = json.load(f)

    train_rxns = set(splits["train_id"])
    val_rxns   = set(splits["val_id"])
    test_rxns  = set(splits["test_id_same_family"])
    ood_rxns   = set(splits["test_ood_family"])

    FORCES_KEY = "wB97x_6-31G(d).forces"

    split_frames:  dict[str, list[int]]   = defaultdict(list)
    split_f_rms:   dict[str, list[float]] = defaultdict(list)

    with h5py.File(a.h5, "r") as f:
        for split_grp_name in f.keys():
            grp = f[split_grp_name]
            if not hasattr(grp, "items"):
                continue
            for formula, fgrp in grp.items():
                for rxn_id, rgrp in fgrp.items():
                    if "positions" not in rgrp or FORCES_KEY not in rgrp:
                        continue
                    n = rgrp["positions"].shape[0]
                    forces = rgrp[FORCES_KEY][()]  # (n_frames, n_atoms, 3)
                    f_rms = float(np.sqrt((forces ** 2).mean()))

                    if rxn_id in train_rxns:
                        label = "train"
                    elif rxn_id in val_rxns:
                        label = "val"
                    elif rxn_id in test_rxns:
                        label = "test_id"
                    elif rxn_id in ood_rxns:
                        label = "test_ood"
                    else:
                        label = "unassigned"

                    split_frames[label].append(n)
                    split_f_rms[label].append(f_rms)

    print("\n=== Frame count statistics per split ===")
    for label in ["train", "val", "test_id", "test_ood", "unassigned"]:
        fc = split_frames.get(label, [])
        if not fc:
            continue
        print(f"  {label:12s}: {len(fc):5d} reactions, "
              f"{sum(fc):7d} total frames, "
              f"median={np.median(fc):.1f}, min={min(fc)}, max={max(fc)}")

    print("\n=== Force RMS statistics per split ===")
    for label in ["train", "val", "test_id", "test_ood"]:
        fr = split_f_rms.get(label, [])
        if not fr:
            continue
        print(f"  {label:12s}: median={np.median(fr):.4f}, "
              f"IQR=[{np.percentile(fr,25):.4f}, {np.percentile(fr,75):.4f}]")

    # Imbalance check
    tf = sum(split_frames.get("train", []))
    vf = sum(split_frames.get("val",   []))
    if vf > 0 and tf < 0.8 * vf:
        print(f"\n[WARNING] Train has {tf} frames vs val {vf} "
              f"({100*tf/vf:.1f}% ratio). This is the known pilot imbalance.")
        print("Possible causes:")
        print("  (a) Train reactions have genuinely shorter IRC paths.")
        print("  (b) Sampling was done per-split not per-reaction (bug).")
        print("  (c) Family split assigned long-path families to val.")
        print("Check median IRC path length per split above.")
        print("If median train path < median val path: cause (a) or (c).")
        print("If medians are equal but totals differ: cause (b).")
    else:
        print(f"\n[OK] Frame balance train={tf} val={vf}")


if __name__ == "__main__":
    main()
