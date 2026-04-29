"""tests/test_coulomb_gradient.py

Hard scientific guardrail: shielded Coulomb energy must be correctly
differentiated so that  F_i = -dE_coul/dR_i  holds to numerical precision.

Tests:
  1. Autograd forces vs central finite differences (double precision, tight tol)
  2. Total force = 0 (Newton's 3rd law / translational symmetry)
  3. Sign of energy for same-sign charges (repulsion must be positive)
  4. Full model API forces consistent with FD (float32, looser tol)

Using double precision for tests 1-3 lets us use a step h=1e-5 Å and demand
  max abs error < 1e-5 eV/Å  and  max rel error < 1e-4
which is tight enough to catch:
  - sign flip in the gradient
  - detach() on R before computing grad
  - missing requires_grad=True
  - incorrect retain_graph usage that frees the graph early
  - accidental non-conservative computation (hand-coded forces)
"""

import pytest
import torch

from ralrc.model_clean import ChargeAwarePotentialClean, shielded_coulomb_energy

# ---------------------------------------------------------------------------
# Shared toy system — 4 atoms, deterministic, no atoms closer than 1.2 Å
# ---------------------------------------------------------------------------

def _toy_system_f64():
    """4-atom geometry in double precision."""
    R = torch.tensor([
        [0.00, 0.00, 0.00],
        [1.50, 0.00, 0.00],
        [0.75, 1.30, 0.00],
        [0.75, 0.43, 1.20],
    ], dtype=torch.float64)
    Z = torch.tensor([1, 6, 7, 8], dtype=torch.long)
    q = torch.tensor([0.30, -0.20, 0.10, -0.20], dtype=torch.float64)
    shield = torch.zeros(119, 119, dtype=torch.float64)
    return q, R, Z, shield


@pytest.fixture(scope="module")
def toy_f64():
    return _toy_system_f64()


# ---------------------------------------------------------------------------
# Test 1: autograd == central-FD, double precision
# ---------------------------------------------------------------------------

def test_coulomb_autograd_vs_fd(toy_f64):
    """
    shielded_coulomb_energy autograd forces must match central finite-difference
    forces to within 1e-5 eV/Å absolute and 1e-4 relative.
    """
    q, R_base, Z, shield = toy_f64
    h = 1e-5  # step in Angstrom

    R = R_base.clone().requires_grad_(True)
    E = shielded_coulomb_energy(q, R, Z, shield)
    F_auto = -torch.autograd.grad(E, R)[0]

    F_fd = torch.zeros_like(R_base)
    for i in range(R_base.shape[0]):
        for d in range(3):
            Rfwd = R_base.clone(); Rfwd[i, d] += h
            Rbwd = R_base.clone(); Rbwd[i, d] -= h
            E_fwd = shielded_coulomb_energy(q, Rfwd, Z, shield)
            E_bwd = shielded_coulomb_energy(q, Rbwd, Z, shield)
            F_fd[i, d] = -((E_fwd - E_bwd) / (2 * h))

    abs_err = (F_auto - F_fd).abs()
    max_abs  = abs_err.max().item()
    denom    = F_fd.abs().clamp(min=1e-10)
    max_rel  = (abs_err / denom).max().item()

    assert max_abs < 1e-5, (
        f"Coulomb autograd max abs error {max_abs:.3e} eV/Å  (limit 1e-5)\n"
        f"F_auto:\n{F_auto}\nF_fd:\n{F_fd}"
    )
    assert max_rel < 1e-4, (
        f"Coulomb autograd max rel error {max_rel:.3e}  (limit 1e-4)\n"
        f"F_auto:\n{F_auto}\nF_fd:\n{F_fd}"
    )


# ---------------------------------------------------------------------------
# Test 2: Newton's 3rd law — total force must be zero
# ---------------------------------------------------------------------------

def test_coulomb_total_force_zero(toy_f64):
    """
    sum_i F_i = 0 for any conservative pairwise potential.
    Failure means the forces are not the true gradient of a scalar potential.
    """
    q, R_base, Z, shield = toy_f64
    R = R_base.clone().requires_grad_(True)
    E = shielded_coulomb_energy(q, R, Z, shield)
    F = -torch.autograd.grad(E, R)[0]

    total = F.sum(dim=0)
    assert total.abs().max().item() < 1e-10, (
        f"Total Coulomb force is not zero (non-conservative): {total.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 3: sign check — same-sign charges must repel (E > 0)
# ---------------------------------------------------------------------------

def test_coulomb_sign_same_charges():
    """
    Two +1 charges 2 Å apart must give E ≈ +K_E / 2 ≈ +7.20 eV.
    Catches sign-flip bugs in K_E or the q_i*q_j product.
    """
    K_E = 14.3996  # must match model_clean.py
    r = 2.0
    q = torch.tensor([1.0, 1.0], dtype=torch.float64)
    R = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
    Z = torch.tensor([1, 1], dtype=torch.long)
    shield = torch.zeros(119, 119, dtype=torch.float64)

    E = shielded_coulomb_energy(q, R, Z, shield)
    expected = K_E / r  # unshielded limit (shield=0 → softplus(0)≈0.693; small correction)

    assert E.item() > 0, (
        f"Same-sign charges must repel (E > 0), got E={E.item():.4f} eV"
    )
    # softplus(0) ≈ 0.693, so effective r = sqrt(4 + 0.48) ≈ 2.12, E ≈ K_E/2.12 ≈ 6.79
    assert 5.0 < E.item() < 10.0, (
        f"Coulomb energy magnitude implausible: {E.item():.4f} eV "
        f"(expected ~6-8 eV for two +1 charges 2 Å apart with softplus shield)"
    )


# ---------------------------------------------------------------------------
# Test 4: opposite-sign charges must attract (E < 0)
# ---------------------------------------------------------------------------

def test_coulomb_sign_opposite_charges():
    """E must be negative for +/- charge pair."""
    q = torch.tensor([1.0, -1.0], dtype=torch.float64)
    R = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    Z = torch.tensor([1, 8], dtype=torch.long)
    shield = torch.zeros(119, 119, dtype=torch.float64)

    E = shielded_coulomb_energy(q, R, Z, shield)
    assert E.item() < 0, (
        f"Opposite-sign charges must attract (E < 0), got E={E.item():.4f} eV"
    )


# ---------------------------------------------------------------------------
# Test 5: full model API — forces consistent with FD (float32)
# ---------------------------------------------------------------------------

def test_model_forces_fd_consistency_coulomb():
    """
    ChargeAwarePotentialClean (use_coulomb=True) forces must agree with FD
    to within 5e-3 eV/Å in float32.  Tests the full autograd path including
    the charge head and local energy head.
    """
    torch.manual_seed(7)
    model = ChargeAwarePotentialClean(hidden=16, use_coulomb=True)
    model.eval()

    Z = torch.tensor([1, 6, 7, 8], dtype=torch.long)
    R_base = torch.tensor([
        [0.00, 0.00, 0.00],
        [1.50, 0.00, 0.00],
        [0.75, 1.30, 0.00],
        [0.75, 0.43, 1.20],
    ], dtype=torch.float32)
    Q = torch.tensor(0, dtype=torch.long)
    S = torch.tensor(1, dtype=torch.long)

    eps = 1e-3
    F_fd = torch.zeros_like(R_base)
    for i in range(R_base.shape[0]):
        for d in range(3):
            Rfwd = R_base.clone(); Rfwd[i, d] += eps
            Rbwd = R_base.clone(); Rbwd[i, d] -= eps
            with torch.no_grad():
                E_fwd = model.forward_energy(Z, Rfwd, Q, S)[0]
                E_bwd = model.forward_energy(Z, Rbwd, Q, S)[0]
            F_fd[i, d] = -((E_fwd - E_bwd) / (2 * eps))

    out = model.forward(Z, R_base, Q, S, compute_forces=True)
    F_auto = out["forces"].detach()

    max_err = (F_auto - F_fd).abs().max().item()
    assert max_err < 5e-3, (
        f"Model (Coulomb=True) force FD mismatch: {max_err:.3e} eV/Å  (limit 5e-3)\n"
        f"F_auto:\n{F_auto}\nF_fd:\n{F_fd}"
    )


def test_model_forces_fd_consistency_no_coulomb():
    """Same test with use_coulomb=False (local-only model)."""
    torch.manual_seed(7)
    model = ChargeAwarePotentialClean(hidden=16, use_coulomb=False)
    model.eval()

    Z = torch.tensor([1, 6, 7, 8], dtype=torch.long)
    R_base = torch.tensor([
        [0.00, 0.00, 0.00],
        [1.50, 0.00, 0.00],
        [0.75, 1.30, 0.00],
        [0.75, 0.43, 1.20],
    ], dtype=torch.float32)
    Q = torch.tensor(0, dtype=torch.long)
    S = torch.tensor(1, dtype=torch.long)

    eps = 1e-3
    F_fd = torch.zeros_like(R_base)
    for i in range(R_base.shape[0]):
        for d in range(3):
            Rfwd = R_base.clone(); Rfwd[i, d] += eps
            Rbwd = R_base.clone(); Rbwd[i, d] -= eps
            with torch.no_grad():
                E_fwd = model.forward_energy(Z, Rfwd, Q, S)[0]
                E_bwd = model.forward_energy(Z, Rbwd, Q, S)[0]
            F_fd[i, d] = -((E_fwd - E_bwd) / (2 * eps))

    out = model.forward(Z, R_base, Q, S, compute_forces=True)
    F_auto = out["forces"].detach()

    max_err = (F_auto - F_fd).abs().max().item()
    assert max_err < 5e-3, (
        f"Model (Coulomb=False) force FD mismatch: {max_err:.3e} eV/Å  (limit 5e-3)\n"
        f"F_auto:\n{F_auto}\nF_fd:\n{F_fd}"
    )
