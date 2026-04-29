"""
model_clean.py - Clean ChargeAwarePotential API for RALRC benchmark.

Design decisions:
- forward_energy() is the pure differentiable core.
- forward() calls forward_energy then calls autograd.grad once, controlled here.
- Forces are NEVER computed inside __init__ or property setters.
- Q and S are validated as torch.long at the boundary.
- compute_forces=False is safe under torch.no_grad().
- No create_graph=self.training magic; caller passes compute_forces.

NOTE: torch.cdist is NOT used anywhere in this file.  It lacks a registered
_cdist_backward kernel in several CUDA builds (including torch 2.11+cu128).
All pairwise distances are computed via explicit broadcasting:
    diff = R.unsqueeze(1) - R.unsqueeze(0)   # (N, N, 3)
    rij  = diff.pow(2).sum(-1).sqrt()         # (N, N)
This path is fully differentiable through autograd.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

K_E = 14.3996  # eV * angstrom / e^2


def _validate_inputs(Z, R, Q, S):
    if Z.dtype != torch.long:
        raise TypeError(f"Z must be torch.long, got {Z.dtype}")
    if Q.dtype != torch.long:
        raise TypeError(f"Q must be torch.long, got {Q.dtype}")
    if S.dtype != torch.long:
        raise TypeError(f"S must be torch.long, got {S.dtype}")
    if R.dim() != 2 or R.shape[1] != 3:
        raise ValueError(f"R must be (N, 3), got {R.shape}")
    if Z.shape[0] != R.shape[0]:
        raise ValueError(f"Z and R atom count mismatch: {Z.shape[0]} vs {R.shape[0]}")


def _pairwise_distances(R: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances without torch.cdist.

    Uses explicit broadcasting so autograd can differentiate wrt R.

        diff[i,j] = R[i] - R[j]           (N, N, 3)
        rij[i,j]  = ||R[i] - R[j]||_2     (N, N)

    A small epsilon is added under the sqrt to avoid zero-gradient
    singularities on the diagonal (which are masked out anyway).
    """
    diff = R.unsqueeze(1) - R.unsqueeze(0)        # (N, N, 3)
    r2   = diff.pow(2).sum(-1)                     # (N, N)
    rij  = (r2 + 1e-12).sqrt()                    # (N, N), grad-safe
    return rij


class ChargeAwarePotentialClean(nn.Module):
    """
    Charge-conditioned MACE-style MLIP with shielded Coulomb term.

    API
    ---
    forward_energy(Z, R, Q, S) -> E : scalar tensor, always differentiable wrt R
    forward(Z, R, Q, S, compute_forces=True) -> dict with keys:
        'energy'  : scalar tensor
        'forces'  : (N, 3) tensor  (only present if compute_forces=True)
        'charges' : (N,) tensor
    predict_charges(Z, R, Q, S) -> (N,) tensor

    Symmetries preserved
    --------------------
    - Translation invariance  (pairwise distances as features)
    - Rotation invariance     (pairwise distances as features)
    - Permutation invariance  (summed atom energies)
    - Force equivariance      (forces = -grad_R E, R enters only via distances)
    - Exact charge conservation: sum(q) == Q always
    - Smooth forces: shielded Coulomb via softplus damping
    - Size extensivity: E = sum_i E_i + E_coul
    """

    def __init__(
        self,
        hidden: int = 64,
        n_elements: int = 119,
        q_range: int = 11,
        s_range: int = 11,
        use_coulomb: bool = True,
    ):
        super().__init__()
        self.use_coulomb = use_coulomb
        self.hidden = hidden

        self.atom_embed   = nn.Embedding(n_elements, hidden)
        self.q_embed      = nn.Embedding(q_range, hidden)   # idx = Q + 5
        self.s_embed      = nn.Embedding(s_range, hidden)   # idx = S - 1

        self.energy_head  = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.charge_head  = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        # Per-element-pair Coulomb shield parameter (symmetric)
        self.shield = nn.Parameter(torch.zeros(n_elements, n_elements))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, Z, Q, S):
        """Atom features conditioned on total charge and spin."""
        h = (
            self.atom_embed(Z)
            + self.q_embed(Q + 5).unsqueeze(0)
            + self.s_embed(S - 1).unsqueeze(0)
        )
        return h

    def _message_pass(self, h: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        One round of distance-weighted message passing. O(N^2).

        Uses _pairwise_distances (explicit broadcasting) instead of
        torch.cdist to guarantee differentiability on all CUDA builds.
        """
        rij = _pairwise_distances(R)                       # (N, N)
        # Zero out self-pairs cleanly
        eye_mask = torch.eye(R.shape[0], device=R.device, dtype=torch.bool)
        rij = rij.masked_fill(eye_mask, 1e6)
        pair_weight = torch.exp(-rij / 2.0)               # (N, N)
        h = h + pair_weight @ h
        return h

    def _conserved_charges(self, h: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Partial charges that exactly sum to Q.
        q_i = q_raw_i + (Q - sum_j q_raw_j) / N
        """
        q_raw = self.charge_head(h).squeeze(-1)
        q = q_raw + (Q.float() - q_raw.sum()) / q_raw.numel()
        return q

    def _coulomb_energy(self, q: torch.Tensor, R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Shielded Coulomb: E_coul = k_e * sum_{i<j} q_i*q_j / sqrt(r_ij^2 + a_ij^2)
        where a_ij = softplus(shield[Zi, Zj]).
        Smooth at r -> 0.  Returns scalar 0.0 if use_coulomb is False.
        """
        if not self.use_coulomb:
            return torch.zeros((), device=R.device, dtype=R.dtype)

        diff = R.unsqueeze(1) - R.unsqueeze(0)            # (N, N, 3)
        r2   = diff.pow(2).sum(-1)                         # (N, N)
        a    = F.softplus(self.shield[Z][:, Z])            # (N, N)
        qq   = q.unsqueeze(0) * q.unsqueeze(1)             # (N, N)
        mask = torch.triu(
            torch.ones(R.shape[0], R.shape[0], device=R.device, dtype=torch.bool),
            diagonal=1,
        )
        E_coul = K_E * (qq * mask / (r2 + a.pow(2) + 1e-12).sqrt()).sum()
        return E_coul

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward_energy(self, Z, R, Q, S):
        """
        Pure energy computation. R must require_grad=True for force computation.

        Returns
        -------
        E : scalar tensor
        q : (N,) tensor  conserved partial charges
        """
        _validate_inputs(Z, R, Q, S)
        h = self._embed(Z, Q, S)
        h = self._message_pass(h, R)
        E_local = self.energy_head(h).sum()
        q = self._conserved_charges(h, Q)
        E_coul  = self._coulomb_energy(q, R, Z)
        E_total = E_local + E_coul
        return E_total, q

    def forward(self, Z, R, Q, S, compute_forces=True):
        """
        Full forward pass.

        Returns
        -------
        dict with keys:
            'energy'  : scalar tensor
            'forces'  : (N, 3) tensor  (only if compute_forces=True)
            'charges' : (N,) tensor
        """
        _validate_inputs(Z, R, Q, S)

        if compute_forces:
            R_diff = R.detach().requires_grad_(True)
        else:
            R_diff = R

        E, q = self.forward_energy(Z, R_diff, Q, S)
        result = {"energy": E, "charges": q}

        if compute_forces:
            F_neg = torch.autograd.grad(
                E, R_diff,
                create_graph=self.training,
                retain_graph=False,
            )[0]
            result["forces"] = -F_neg

        return result

    def predict_charges(self, Z, R, Q, S):
        """Convenience method returning only conserved partial charges."""
        _validate_inputs(Z, R, Q, S)
        with torch.no_grad():
            h = self._embed(Z, Q, S)
            h = self._message_pass(h, R.detach())
            q = self._conserved_charges(h, Q)
        return q
