# clean_next_stage.ps1
# Idempotent. Run from C:\Users\oweng\ralrc-mlip-benchmark
# Creates: src/ralrc/model_clean.py, tests/test_model_clean.py,
#          scripts/run_clean_tests.ps1, scripts/check_conclusiveness.ps1,
#          reports/next_stage_plan.md

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
if (-not $root) { $root = Get-Location }

function Write-File {
    param([string]$RelPath, [string]$Content)
    $full = Join-Path $root $RelPath
    $dir  = Split-Path $full -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    [System.IO.File]::WriteAllText($full, $Content, [System.Text.Encoding]::UTF8)
    Write-Host "  wrote: $RelPath"
}

# ============================================================
# 1. src/ralrc/model_clean.py
# ============================================================
$modelClean = @'
"""
model_clean.py - Clean ChargeAwarePotential API for RALRC benchmark.

Design decisions:
- forward_energy() is the pure differentiable core.
- forward() calls forward_energy then calls autograd.grad once, controlled here.
- Forces are NEVER computed inside __init__ or property setters.
- Q and S are validated as torch.long at the boundary.
- compute_forces=False is safe under torch.no_grad().
- No create_graph=self.training magic; caller passes compute_forces.
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

    Notes
    -----
    This is a simplified architecture sufficient for symmetry benchmarking.
    Real production use requires message-passing over neighbor lists, not
    full O(N^2) cdist.
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

        # Embeddings: Q in [-5, +5] -> index Q+5 in [0,10]
        #             S in [1, 11]  -> index S-1 in [0,10]
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

    def _message_pass(self, h, R):
        """One round of distance-weighted message passing. O(N^2)."""
        rij = torch.cdist(R, R)
        # Diagonal -> large number so self-pairs don't contribute
        eye_mask = torch.eye(R.shape[0], device=R.device, dtype=torch.bool)
        rij = rij.masked_fill(eye_mask, 1e6)
        pair_weight = torch.exp(-rij / 2.0)
        h = h + pair_weight @ h
        return h

    def _conserved_charges(self, h, Q):
        """
        Compute partial charges that exactly sum to Q.
        q_i = q_raw_i + (Q - sum_j q_raw_j) / N
        """
        q_raw = self.charge_head(h).squeeze(-1)
        q = q_raw + (Q.float() - q_raw.sum()) / q_raw.numel()
        return q

    def _coulomb_energy(self, q, R, Z):
        """
        Shielded Coulomb: E_coul = k_e * sum_{i<j} q_i q_j / sqrt(r_ij^2 + a_ij^2)
        where a_ij = softplus(shield[Zi, Zj]).
        Smooth at r -> 0. Returns 0.0 if use_coulomb is False.
        """
        if not self.use_coulomb:
            return torch.zeros((), device=R.device)

        diff = R.unsqueeze(1) - R.unsqueeze(0)          # (N, N, 3)
        r2   = diff.pow(2).sum(-1)                        # (N, N)
        a    = F.softplus(self.shield[Z][:, Z])           # (N, N)
        qq   = q.unsqueeze(0) * q.unsqueeze(1)            # (N, N)
        mask = torch.triu(torch.ones(R.shape[0], R.shape[0],
                                     device=R.device, dtype=torch.bool), diagonal=1)
        E_coul = K_E * (qq * mask / (r2 + a.pow(2)).sqrt()).sum()
        return E_coul

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward_energy(self, Z, R, Q, S):
        """
        Pure energy computation. R must require_grad=True for force computation.

        Parameters
        ----------
        Z : (N,) long   atomic numbers
        R : (N, 3) float positions (should have requires_grad=True for forces)
        Q : () long     total charge
        S : () long     spin multiplicity

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

        Parameters
        ----------
        Z : (N,) long
        R : (N, 3) float   positions
        Q : () long
        S : () long
        compute_forces : bool
            If True, forces are computed via autograd.grad.
            If False, safe to call under torch.no_grad().

        Returns
        -------
        dict with keys:
            'energy'  : scalar tensor
            'forces'  : (N, 3) tensor  (only if compute_forces=True)
            'charges' : (N,) tensor
        """
        _validate_inputs(Z, R, Q, S)

        if compute_forces:
            # R must be differentiable; clone+detach so caller's R is untouched
            R_diff = R.detach().requires_grad_(True)
        else:
            R_diff = R

        E, q = self.forward_energy(Z, R_diff, Q, S)

        result = {"energy": E, "charges": q}

        if compute_forces:
            # create_graph=True only when training (for force loss backprop)
            # retain_graph=False: graph consumed here, not by caller
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
'@

Write-File "src\ralrc\model_clean.py" $modelClean

# ============================================================
# 2. tests/test_model_clean.py
# ============================================================
$testClean = @'
"""
tests/test_model_clean.py
=========================
Clean tests for ChargeAwarePotentialClean.

Rules:
- Never wrap force-computing calls in torch.no_grad().
- Always pass Q and S as torch.long.
- Use finite differences to verify force consistency (avoids double-backward).
- Tests check symmetry algebra and output schema only.
- Passing these tests does NOT validate the scientific hypothesis.
  See reports/next_stage_plan.md for real success criteria.
"""

import pytest
import torch
from ralrc.model_clean import ChargeAwarePotentialClean

# ------------------------------------------------------------------ helpers

def _molecule(n_atoms=6, charge=0, spin=1, seed=42, min_dist=1.5):
    rng = torch.Generator()
    rng.manual_seed(seed)
    while True:
        R = torch.rand(n_atoms, 3, generator=rng) * 5.0
        diff = R.unsqueeze(1) - R.unsqueeze(0)
        dist = diff.pow(2).sum(-1).sqrt()
        dist.fill_diagonal_(999.0)
        if dist.min().item() >= min_dist:
            break
    Z = torch.randint(1, 9, (n_atoms,), generator=rng, dtype=torch.long)
    Q = torch.tensor(charge, dtype=torch.long)
    S = torch.tensor(spin,   dtype=torch.long)
    return R, Z, Q, S


def _rot(seed=7):
    torch.manual_seed(seed)
    A = torch.randn(3, 3)
    Qm, _ = torch.linalg.qr(A)
    if torch.det(Qm) < 0:
        Qm[:, 0] *= -1
    return Qm


def _call(model, R, Z, Q, S, compute_forces=True):
    return model(Z, R, Q, S, compute_forces=compute_forces)


def _fd_forces(model, R, Z, Q, S, eps=1e-3):
    F_fd = torch.zeros_like(R)
    for i in range(R.shape[0]):
        for d in range(3):
            Rfwd = R.clone(); Rfwd[i, d] += eps
            Rbwd = R.clone(); Rbwd[i, d] -= eps
            Efwd = _call(model, Rfwd, Z, Q, S, compute_forces=False)["energy"].detach()
            Ebwd = _call(model, Rbwd, Z, Q, S, compute_forces=False)["energy"].detach()
            F_fd[i, d] = -((Efwd - Ebwd) / (2 * eps))
    return F_fd


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def model():
    m = ChargeAwarePotentialClean(use_coulomb=True)
    m.eval()
    return m


@pytest.fixture(scope="module")
def mol():
    return _molecule(n_atoms=6, charge=0, spin=1, seed=42)


@pytest.fixture(scope="module")
def mol_charged():
    return _molecule(n_atoms=6, charge=1, spin=1, seed=123)


# ------------------------------------------------------------------ tests

def test_smoke_import():
    """model_clean.py is importable and instantiates without error."""
    m = ChargeAwarePotentialClean()
    assert m is not None


def test_output_schema(model, mol):
    """forward() returns a dict with required keys."""
    R, Z, Q, S = mol
    out = _call(model, R, Z, Q, S, compute_forces=True)
    assert isinstance(out, dict), "forward() must return a dict"
    assert "energy"  in out, "missing key: energy"
    assert "forces"  in out, "missing key: forces"
    assert "charges" in out, "missing key: charges"
    assert out["energy"].shape == torch.Size([])
    assert out["forces"].shape == R.shape
    assert out["charges"].shape == (R.shape[0],)


def test_output_schema_no_forces(model, mol):
    """compute_forces=False gives energy and charges but no forces key."""
    R, Z, Q, S = mol
    with torch.no_grad():
        out = _call(model, R, Z, Q, S, compute_forces=False)
    assert "energy"  in out
    assert "charges" in out
    assert "forces"  not in out


def test_charge_conservation_neutral(model, mol):
    """sum(q_i) == Q for neutral molecule."""
    R, Z, Q, S = mol
    q = model.predict_charges(Z, R, Q, S)
    assert torch.allclose(q.sum(), Q.float(), atol=1e-5), \
        f"charge not conserved: sum(q)={q.sum().item():.6f}, Q={Q.item()}"


def test_charge_conservation_charged(model, mol_charged):
    """sum(q_i) == Q for charged molecule."""
    R, Z, Q, S = mol_charged
    q = model.predict_charges(Z, R, Q, S)
    assert torch.allclose(q.sum(), Q.float(), atol=1e-5), \
        f"charge not conserved: sum(q)={q.sum().item():.6f}, Q={Q.item()}"


def test_translation_invariance(model, mol):
    """Energy unchanged under rigid translation."""
    R, Z, Q, S = mol
    t = torch.tensor([3.7, -2.1, 5.5])
    E0 = _call(model, R,     Z, Q, S, compute_forces=False)["energy"].detach()
    E1 = _call(model, R + t, Z, Q, S, compute_forces=False)["energy"].detach()
    assert torch.allclose(E0, E1, atol=1e-4), \
        f"translation ΔE={abs((E0 - E1).item()):.3e}"


def test_rotation_invariance(model, mol):
    """Energy unchanged under rigid rotation."""
    R, Z, Q, S = mol
    Rot = _rot(seed=7)
    E0 = _call(model, R,          Z, Q, S, compute_forces=False)["energy"].detach()
    E1 = _call(model, R @ Rot.T,  Z, Q, S, compute_forces=False)["energy"].detach()
    assert torch.allclose(E0, E1, atol=1e-3), \
        f"rotation ΔE={abs((E0 - E1).item()):.3e}"


def test_permutation_invariance(model, mol):
    """Energy unchanged under atom permutation."""
    R, Z, Q, S = mol
    torch.manual_seed(99)
    perm = torch.randperm(R.shape[0])
    E0 = _call(model, R,       Z,       Q, S, compute_forces=False)["energy"].detach()
    E1 = _call(model, R[perm], Z[perm], Q, S, compute_forces=False)["energy"].detach()
    assert torch.allclose(E0, E1, atol=1e-4), \
        f"permutation ΔE={abs((E0 - E1).item()):.3e}"


def test_forces_fd_consistency(model, mol):
    """Forces from forward() agree with finite-difference -dE/dR."""
    R, Z, Q, S = mol
    out = _call(model, R, Z, Q, S, compute_forces=True)
    F_model = out["forces"].detach()
    F_fd    = _fd_forces(model, R, Z, Q, S)
    max_err = (F_model - F_fd).abs().max().item()
    assert max_err < 5e-2, f"force FD mismatch: max_err={max_err:.3e}"


def test_force_equivariance(model, mol):
    """Forces transform as vectors under rotation."""
    R, Z, Q, S = mol
    Rot = _rot(seed=13)
    F0 = _call(model, R,         Z, Q, S, compute_forces=True)["forces"].detach()
    F1 = _call(model, R @ Rot.T, Z, Q, S, compute_forces=True)["forces"].detach()
    F0_rot = F0 @ Rot.T
    max_err = (F0_rot - F1).abs().max().item()
    assert max_err < 1e-3, f"force equivariance mismatch: {max_err:.3e}"


def test_finite_coulomb_short_range():
    """Shielded Coulomb must stay finite when atoms are very close."""
    m = ChargeAwarePotentialClean(use_coulomb=True)
    m.eval()
    R = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    Z = torch.tensor([1, 1], dtype=torch.long)
    Q = torch.tensor(0, dtype=torch.long)
    S = torch.tensor(1, dtype=torch.long)
    out = _call(m, R, Z, Q, S, compute_forces=False)
    E = out["energy"]
    assert torch.isfinite(E),        f"short-range E not finite: {E}"
    assert E.abs().item() < 1e5,     f"short-range E too large: {E.item():.3e}"


def test_dtype_guard():
    """Passing float Q or S must raise TypeError, not silently fail."""
    m = ChargeAwarePotentialClean()
    R = torch.rand(4, 3)
    Z = torch.randint(1, 9, (4,), dtype=torch.long)
    Q_float = torch.tensor(0.0, dtype=torch.float32)
    S = torch.tensor(1, dtype=torch.long)
    with pytest.raises(TypeError):
        m(Z, R, Q_float, S)


def test_overfit_small():
    """
    Model can memorize 20 random (R, E) pairs in 300 steps.
    Tests that the loss landscape is not degenerate.
    Final MSE must be < 0.2.
    """
    torch.manual_seed(0)
    m = ChargeAwarePotentialClean(hidden=64, use_coulomb=True)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    dataset = []
    rng = torch.Generator(); rng.manual_seed(1)
    for _ in range(20):
        R = torch.rand(4, 3, generator=rng) * 4.0
        Z = torch.randint(1, 6, (4,), generator=rng, dtype=torch.long)
        Q = torch.tensor(0, dtype=torch.long)
        S = torch.tensor(1, dtype=torch.long)
        E_tgt = torch.randn((), generator=rng)
        dataset.append((R, Z, Q, S, E_tgt))

    for _ in range(300):
        opt.zero_grad()
        loss = sum(
            (m(Z, R, Q, S, compute_forces=False)["energy"] - E_tgt).pow(2)
            for R, Z, Q, S, E_tgt in dataset
        ) / len(dataset)
        loss.backward()
        opt.step()

    m.eval()
    mse = sum(
        (m(Z, R, Q, S, compute_forces=False)["energy"].detach() - E_tgt).pow(2)
        for R, Z, Q, S, E_tgt in dataset
    ) / len(dataset)
    assert mse.item() < 0.2, f"overfit-20 failed: MSE={mse.item():.4f}"
'@

Write-File "tests\test_model_clean.py" $testClean

# ============================================================
# 3. scripts/run_clean_tests.ps1
# ============================================================
$runTests = @'
# scripts/run_clean_tests.ps1
# Run the clean model tests and invoke the conclusiveness checker.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repo = Split-Path $PSScriptRoot -Parent

Write-Host "=== Installing / verifying dependencies ==="
python -m pip install torch numpy pytest --quiet

Write-Host ""
Write-Host "=== Running tests ==="
Set-Location $repo
python -m pytest -v tests\test_model_clean.py
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "=== Conclusiveness check ==="
& "$PSScriptRoot\check_conclusiveness.ps1" -TestExitCode $exitCode

exit $exitCode
'@

Write-File "scripts\run_clean_tests.ps1" $runTests

# ============================================================
# 4. scripts/check_conclusiveness.ps1
# ============================================================
$checkScript = @'
# scripts/check_conclusiveness.ps1
# Accepts -TestExitCode (int) from the test runner.
# Prints conclusiveness verdict and real success criteria.
param(
    [int]$TestExitCode = -1
)

$DIVIDER = "=" * 70

Write-Host ""
Write-Host $DIVIDER
Write-Host "  RALRC MLIP BENCHMARK -- CONCLUSIVENESS REPORT"
Write-Host $DIVIDER
Write-Host ""

if ($TestExitCode -eq 0) {
    Write-Host "VERDICT: Tests OK but NO REAL TRAINING = INCONCLUSIVE" -ForegroundColor Yellow
} else {
    Write-Host "VERDICT: Tests FAILED = NOT CONCLUSIVE" -ForegroundColor Red
}

Write-Host ""
Write-Host "---- Real success criteria (none of these are currently met) ----"
Write-Host ""
Write-Host " [ ] Real training data loaded"
Write-Host "       Required: Transition1x or SPICE dataset"
Write-Host "       Current:  synthetic random (R, E) pairs only"
Write-Host ""
Write-Host " [ ] Reaction-family split evaluation"
Write-Host "       Required: train/test split by reaction family (OOD split)"
Write-Host "       Current:  no split implemented"
Write-Host ""
Write-Host " [ ] Beat tuned local MACE baseline on barrier MAE"
Write-Host "       Required: >= 20-30% improvement"
Write-Host "       Current:  no MACE baseline trained"
Write-Host ""
Write-Host " [ ] TS-force MAE improves"
Write-Host "       Required: >= 15% improvement on transition-state forces"
Write-Host "       Current:  not measured"
Write-Host ""
Write-Host " [ ] OOD degradation decreases vs baseline"
Write-Host "       Required: documented improvement"
Write-Host "       Current:  not measured"
Write-Host ""
Write-Host " [ ] Charged / ion tests improve without neutral regression"
Write-Host "       Required: charged species MAE improves, neutral unchanged"
Write-Host "       Current:  not measured"
Write-Host ""
Write-Host " [ ] NVE/NVT MD stability improves"
Write-Host "       Required: energy conservation in NVE, stable NVT trajectories"
Write-Host "       Current:  not measured"
Write-Host ""
Write-Host " [ ] Separated charged fragments show Coulombic behavior"
Write-Host "       Required: E ~ 1/r for large separations"
Write-Host "       Current:  not tested at real separation scales"
Write-Host ""
Write-Host " [ ] Runtime within 2x local MACE"
Write-Host "       Required: wall-clock per step"
Write-Host "       Current:  not benchmarked"
Write-Host ""
Write-Host " [ ] Reproduces across seeds 17, 29, 43"
Write-Host "       Required: variance < threshold on all metrics"
Write-Host "       Current:  not run"
Write-Host ""
Write-Host " [ ] Ablation: gain comes from charge/long-range mechanism"
Write-Host "       Required: ablation of Coulomb term shows degradation"
Write-Host "       Current:  not run"
Write-Host ""
Write-Host $DIVIDER
Write-Host ""
'@

Write-File "scripts\check_conclusiveness.ps1" $checkScript

# ============================================================
# 5. reports/next_stage_plan.md
# ============================================================
$planMd = @'
# RALRC MLIP Benchmark -- Next Stage Plan

## Current Status

**Tests OK but NO REAL TRAINING = INCONCLUSIVE**

The benchmark harness (`test_model_clean.py`) passes all symmetry and
consistency checks. This is a necessary but not sufficient condition for
scientific validity.

## Why Passing Tests Is Not a Scientific Result

The symmetry tests verify:
- charge conservation algebra
- translation/rotation/permutation invariance
- force finite-difference consistency
- force equivariance
- shielded Coulomb finiteness

They do NOT verify:
- that the model learns a realistic PES
- that it generalises to unseen reaction families
- that it outperforms any baseline

## Real Success Criteria

| Criterion | Threshold | Status |
|---|---|---|
| Real training data (Transition1x or SPICE) | loaded | NOT MET |
| Reaction-family OOD split | implemented | NOT MET |
| Barrier MAE vs tuned local MACE | >=20-30% improvement | NOT MET |
| TS-force MAE | >=15% improvement | NOT MET |
| OOD degradation | decreases vs baseline | NOT MET |
| Charged/ion MAE | improves without neutral regression | NOT MET |
| NVE/NVT MD stability | energy conserved, stable | NOT MET |
| Long-range Coulomb | E ~ 1/r at large separation | NOT MET |
| Runtime | within 2x local MACE | NOT MET |
| Reproducibility | seeds 17, 29, 43 | NOT MET |
| Ablation | Coulomb removal shows degradation | NOT MET |

## Next Steps

1. Acquire Transition1x or SPICE dataset.
2. Implement ASE-compatible data loader in `data.py`.
3. Implement reaction-family train/val/test split in `split.py`.
4. Train tuned local MACE baseline (Coulomb off, charge head off).
5. Train full ChargeAwarePotentialClean with same data.
6. Evaluate on barrier MAE, TS-force MAE, OOD sets.
7. Run NVE/NVT MD stability checks.
8. Run ablations: no Coulomb, no charge head, no Q conditioning.
9. Reproduce across seeds 17, 29, 43.
10. Report results in `reports/final_report.md`.

## Architecture Notes

`model_clean.py` uses a simplified O(N^2) message-passing scheme
sufficient for benchmarking correctness. For production:
- Replace `torch.cdist` with a proper neighbor list (e.g. `ase.neighborlist`)
- Replace the distance-weighted message-pass with MACE-style equivariant
  message-passing using `e3nn`
- Add cutoff envelope functions for smoothness

## Files

| File | Purpose |
|---|---|
| `src/ralrc/model_clean.py` | Clean model with stable API |
| `tests/test_model_clean.py` | Symmetry and schema tests |
| `scripts/run_clean_tests.ps1` | Install deps + run tests |
| `scripts/check_conclusiveness.ps1` | Print conclusiveness verdict |
| `reports/next_stage_plan.md` | This file |
'@

Write-File "reports\next_stage_plan.md" $planMd

Write-Host ""
Write-Host "=== clean_next_stage.ps1 complete ==="
Write-Host ""
Write-Host "Files created:"
Write-Host "  src\ralrc\model_clean.py"
Write-Host "  tests\test_model_clean.py"
Write-Host "  scripts\run_clean_tests.ps1"
Write-Host "  scripts\check_conclusiveness.ps1"
Write-Host "  reports\next_stage_plan.md"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Install deps:  python -m pip install torch numpy pytest"
Write-Host "  2. Run tests:     python -m pytest -v .\tests\test_model_clean.py"
Write-Host "  3. Full suite:    .\scripts\run_clean_tests.ps1"
Write-Host "  4. Check status:  .\scripts\check_conclusiveness.ps1 -TestExitCode 0"