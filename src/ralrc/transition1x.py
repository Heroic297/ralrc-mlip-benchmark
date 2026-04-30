"""Lazy HDF5 reader/iterator for Transition1x.

Never loads the full file into RAM. Yields one sample dict at a time.
Each sample:
  {
    'z':        np.ndarray (n_atoms,)       int32  atomic numbers
    'pos':      np.ndarray (n_atoms, 3)     float64 positions (Angstrom)
    'energy':   float                        Hartree
    'forces':   np.ndarray (n_atoms, 3)     float64 Hartree/Angstrom
    'formula':  str
    'rxn_id':   str
    'frame_idx': int
    'split':    str  ('train'|'val'|'test'|'data')
  }
Endpoint samples (reactant/product/transition_state) are yielded
when include_endpoints=True; they carry frame_idx=-1 and an extra
'endpoint' key.
"""

from __future__ import annotations

import numpy as np
import h5py
from pathlib import Path
from typing import Iterator, Optional, Sequence

ENERGY_KEY = "wB97x_6-31G(d).energy"
FORCES_KEY = "wB97x_6-31G(d).forces"
ENDPOINTS = ("reactant", "product", "transition_state")
REQUIRED_KEYS = {"atomic_numbers", "positions", ENERGY_KEY, FORCES_KEY}

# Energy unit: Transition1x HDF5 stores energies in eV and forces in eV/Å
# (Schreiner et al. 2022, Scientific Data). Atomic refs from
# fit_atomic_refs.py are also in eV; subtraction is unitless.


def _check_rxn_group(grp: h5py.Group) -> list[str]:
    """Return list of missing required keys for a reaction group."""
    missing = []
    for k in REQUIRED_KEYS:
        if k not in grp:
            missing.append(k)
    return missing


def _sample_from_group(
    grp: h5py.Group,
    formula: str,
    rxn_id: str,
    split: str,
    frame_idx: int,
    endpoint: Optional[str] = None,
) -> dict:
    """Read one frame from an open HDF5 group (no full-array load)."""
    z = grp["atomic_numbers"][()]  # (n_atoms,)
    pos_full = grp["positions"]  # (n_frames, n_atoms, 3)
    e_full = grp[ENERGY_KEY]      # (n_frames,)
    f_full = grp[FORCES_KEY]      # (n_frames, n_atoms, 3)

    pos = pos_full[frame_idx]  # (n_atoms, 3)
    e = float(e_full[frame_idx])
    forces = f_full[frame_idx]  # (n_atoms, 3)

    sample = {
        "z": np.array(z, dtype=np.int32),
        "pos": np.array(pos, dtype=np.float64),
        "energy": e,
        "forces": np.array(forces, dtype=np.float64),
        "formula": formula,
        "rxn_id": rxn_id,
        "frame_idx": frame_idx,
        "split": split,
    }
    if endpoint is not None:
        sample["endpoint"] = endpoint
    return sample


def iter_transition1x(
    h5_path: str | Path,
    splits: Optional[Sequence[str]] = None,
    formulas: Optional[Sequence[str]] = None,
    include_endpoints: bool = False,
    max_reactions: Optional[int] = None,
) -> Iterator[dict]:
    """Iterate over Transition1x frame-by-frame without loading into RAM.

    Parameters
    ----------
    h5_path:
        Path to Transition1x.h5
    splits:
        Subset of root keys to iterate (default: all of train/val/test/data).
    formulas:
        If given, only yield reactions matching these molecular formulas.
    include_endpoints:
        If True, also yield reactant/product/transition_state frames.
    max_reactions:
        Stop after this many reactions (useful for smoke tests).
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r", swmr=False) as f:
        available_splits = [s for s in ("train", "val", "test") if s in f]
        target_splits = splits if splits is not None else available_splits

        rxn_count = 0
        for split in target_splits:
            if split not in f:
                continue
            split_grp = f[split]
            for formula, frml_grp in split_grp.items():
                if formulas is not None and formula not in formulas:
                    continue
                if not isinstance(frml_grp, h5py.Group):
                    continue
                for rxn_id, rxn_grp in frml_grp.items():
                    if not isinstance(rxn_grp, h5py.Group):
                        continue
                    missing = _check_rxn_group(rxn_grp)
                    if missing:
                        # skip silently; summary script will report these
                        continue

                    n_frames = rxn_grp["positions"].shape[0]
                    for fi in range(n_frames):
                        yield _sample_from_group(
                            rxn_grp, formula, rxn_id, split, fi
                        )

                    if include_endpoints:
                        for ep in ENDPOINTS:
                            if ep in rxn_grp:
                                ep_grp = rxn_grp[ep]
                                ep_missing = _check_rxn_group(ep_grp)
                                if not ep_missing:
                                    yield _sample_from_group(
                                        ep_grp, formula, rxn_id, split, 0, endpoint=ep
                                    )

                    rxn_count += 1
                    if max_reactions is not None and rxn_count >= max_reactions:
                        return


class Transition1xDataset:
    """Indexable wrapper around iter_transition1x.

    Builds an in-memory index of (split, formula, rxn_id, frame_idx) tuples
    without loading actual array data; data is loaded on __getitem__.
    """

    def __init__(
        self,
        h5_path: str | Path,
        splits: Optional[Sequence[str]] = None,
        formulas: Optional[Sequence[str]] = None,
        include_endpoints: bool = False,
        atom_refs: Optional[dict] = None,
    ):
        self.h5_path = Path(h5_path)
        self.splits = splits
        self.formulas = formulas
        self.include_endpoints = include_endpoints
        # atom_refs: {Z(int): e_ref_eV(float)}. JSON files use str keys, normalize.
        if atom_refs is None:
            self._atom_refs_ev = None
        else:
            arr = np.zeros(119, dtype=np.float64)
            for k, v in atom_refs.items():
                arr[int(k)] = float(v)
            self._atom_refs_ev = arr
        self._index: list[tuple] = []  # (split, formula, rxn_id, frame_idx, endpoint|None)
        self._build_index()

    def _build_index(self):
        with h5py.File(self.h5_path, "r") as f:
            available = [s for s in ("train", "val", "test") if s in f]
            target_splits = self.splits if self.splits else available
            for split in target_splits:
                if split not in f:
                    continue
                for formula, frml_grp in f[split].items():
                    if self.formulas and formula not in self.formulas:
                        continue
                    if not isinstance(frml_grp, h5py.Group):
                        continue
                    for rxn_id, rxn_grp in frml_grp.items():
                        if not isinstance(rxn_grp, h5py.Group):
                            continue
                        if _check_rxn_group(rxn_grp):
                            continue
                        n = rxn_grp["positions"].shape[0]
                        for fi in range(n):
                            self._index.append((split, formula, rxn_id, fi, None))
                        if self.include_endpoints:
                            for ep in ENDPOINTS:
                                if ep in rxn_grp and not _check_rxn_group(rxn_grp[ep]):
                                    self._index.append((split, formula, rxn_id, 0, ep))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        split, formula, rxn_id, frame_idx, endpoint = self._index[idx]
        with h5py.File(self.h5_path, "r") as f:
            if endpoint is None:
                grp = f[split][formula][rxn_id]
            else:
                grp = f[split][formula][rxn_id][endpoint]
            sample = _sample_from_group(grp, formula, rxn_id, split, frame_idx, endpoint)
        if self._atom_refs_ev is not None:
            z = sample["z"]
            correction_ev = float(self._atom_refs_ev[z].sum())
            sample["energy"] = sample["energy"] - correction_ev
            sample["atom_ref_correction_ev"] = correction_ev
        return sample
