"""Fit per-element atomic reference energies via closed-form least squares.

Solves  E_total ≈ X @ e_ref   (in eV)

  X        : (n_frames, 119)  atom-count matrix per frame
  e_ref    : (119,)            per-element reference energy (eV)
  E_total  : (n_frames,)       DFT total energy (eV)

Operates on TRAIN-split compound keys only. Writes JSON keyed by atomic
number Z (str), values in eV.

Usage:
    python scripts/fit_atomic_refs.py \\
        --h5 data/transition1x.h5 \\
        --splits splits_pilot.json \\
        --out runs/e_ref.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ralrc.transition1x import Transition1xDataset

# Transition1x stores energies in eV (Schreiner et al. 2022); no conversion.
N_ELEMENTS = 119


def _compound_key(entry: tuple) -> str:
    return f"{entry[0]}::{entry[1]}::{entry[2]}"


def fit_atomic_refs(
    h5_path: Path,
    splits_json: Path,
    out_path: Path,
    max_frames: int | None = None,
) -> dict[int, float]:
    splits = json.loads(splits_json.read_text())
    train_keys = set(splits["train_id"])

    print(f"[fit_atomic_refs] Building dataset index from {h5_path}...")
    ds = Transition1xDataset(h5_path, splits=None)
    print(f"[fit_atomic_refs] Index size: {len(ds):,} frames")

    train_indices = [
        i for i, e in enumerate(ds._index) if _compound_key(e) in train_keys
    ]
    if not train_indices:
        raise RuntimeError(
            "[fit_atomic_refs] FATAL: 0 train frames matched. Stale splits?"
        )
    if max_frames is not None and len(train_indices) > max_frames:
        rng = np.random.default_rng(0)
        train_indices = rng.choice(
            train_indices, size=max_frames, replace=False
        ).tolist()

    n = len(train_indices)
    print(f"[fit_atomic_refs] Fitting on {n:,} train frames")

    X = np.zeros((n, N_ELEMENTS), dtype=np.float64)
    y = np.zeros((n,), dtype=np.float64)

    for row, idx in enumerate(train_indices):
        sample = ds[idx]
        z = np.asarray(sample["z"], dtype=np.int64)
        counts = np.bincount(z, minlength=N_ELEMENTS).astype(np.float64)
        X[row] = counts
        y[row] = float(sample["energy"])  # already in eV
        if row % 5000 == 0 and row > 0:
            print(f"  loaded {row:,}/{n:,}")

    print("[fit_atomic_refs] Solving lstsq...")
    e_ref, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ e_ref
    rms = float(np.sqrt(np.mean((pred - y) ** 2)))
    print(
        f"[fit_atomic_refs] Done. rank={rank}  RMS residual={rms:.3f} eV  "
        f"(per-frame mean abs target={float(np.mean(np.abs(y))):.1f} eV)"
    )

    # Only emit elements that actually appeared in training data.
    present = (X.sum(axis=0) > 0)
    refs = {
        int(z): float(e_ref[z]) for z in range(N_ELEMENTS) if present[z]
    }
    print(f"[fit_atomic_refs] {len(refs)} elements fitted: {sorted(refs)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({str(k): v for k, v in refs.items()}, indent=2))
    print(f"[fit_atomic_refs] Wrote {out_path}")
    return refs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True, type=Path)
    p.add_argument("--splits", required=True, type=Path)
    p.add_argument("--out", default=Path("runs/e_ref.json"), type=Path)
    p.add_argument("--max-frames", type=int, default=None,
                   help="If set, subsample TRAIN frames before lstsq.")
    a = p.parse_args()
    fit_atomic_refs(a.h5, a.splits, a.out, a.max_frames)


if __name__ == "__main__":
    main()
