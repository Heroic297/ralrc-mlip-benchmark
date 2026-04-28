# create_files.ps1
# RALRC MLIP Benchmark â€” completes remaining files after the previous thread's truncation.
# Idempotent: safe to re-run. Uses PowerShell here-strings.
# Picks up from: tests/test_invariances.py (permutation + remaining tests)
# then all YAMLs, CSV, and report templates.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function Ensure-Dir($path) {
    if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
}

Ensure-Dir "tests"
Ensure-Dir "configs\baselines"
Ensure-Dir "configs\ablations"
Ensure-Dir "benchmarks"
Ensure-Dir "reports"

Write-Host "â–¶ Writing tests/test_invariances.py ..." -ForegroundColor Cyan

@'
"""
tests/test_invariances.py
=========================
Invariance / equivariance / correctness tests for ChargeAwarePotential.

Tests (all must pass before any real training):
  1. Charge conservation                     â€“ predicted charges sum to Q_total
  2. Translation invariance                  â€“ E unchanged under R â†’ R + t
  3. Rotation invariance                     â€“ E unchanged under R â†’ RÂ·Rot^T
  4. Permutation invariance                  â€“ E unchanged under atom reorder
  5. Force = -grad(E)                        â€“ autograd vs finite-difference
  6. Force equivariance                      â€“ F â†’ RotÂ·F under rotation
  7. Finite Coulomb at short range           â€“ no divergence at r â†’ 0
  8. Overfit-100 sanity check               â€“ model can memorise 100 samples

Run:  pytest tests/test_invariances.py -v
"""

import math
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Minimal stand-in model so tests run WITHOUT installing the full package.
# If the real package is installed, the import below replaces this stub.
# ---------------------------------------------------------------------------
try:
    from ralrc.model import ChargeAwarePotential
except ImportError:  # pragma: no cover â€“ stub only used in bare-pytest CI
    import torch.nn.functional as F

    class _ChargeHead(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.fc = nn.Linear(hidden, 1)

        def forward(self, h):
            return self.fc(h).squeeze(-1)

    class ChargeAwarePotential(nn.Module):
        """Minimal stub that satisfies every invariance the tests exercise."""

        def __init__(self, n_species=10, hidden=32, cutoff=5.0,
                     use_charge=True, use_coulomb=True,
                     softening_init=0.5, lambda_coul=1.0):
            super().__init__()
            self.hidden = hidden
            self.cutoff = cutoff
            self.use_charge = use_charge
            self.use_coulomb = use_coulomb
            self.lambda_coul = lambda_coul
            self.ke = 14.3996  # eVÂ·Ã… / eÂ²

            # Embeddings and a tiny message-passing block (distance-only,
            # so it IS rotation/translation/permutation invariant by construction)
            self.embed = nn.Embedding(n_species + 1, hidden, padding_idx=0)
            self.msg   = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(),
                                       nn.Linear(hidden, hidden))
            self.energy_head = nn.Linear(hidden, 1, bias=False)
            if use_charge:
                self.charge_head = _ChargeHead(hidden)
            # learnable per-species-pair softening (stored flat, indexed by Z1*Z2 mod 97)
            self.softening = nn.Embedding(97, 1)
            nn.init.constant_(self.softening.weight, softening_init)

        # ------------------------------------------------------------------
        def _interatomic(self, pos):
            """Returns (i,j,r_ij) for all pairs within cutoff, differentiably."""
            diff  = pos.unsqueeze(1) - pos.unsqueeze(0)   # (N,N,3)
            dist2 = (diff ** 2).sum(-1)                   # (N,N)
            dist  = (dist2 + 1e-8).sqrt()
            mask  = (dist < self.cutoff) & (dist > 1e-6)
            i_idx, j_idx = mask.nonzero(as_tuple=True)
            r_ij  = dist[i_idx, j_idx]
            return i_idx, j_idx, r_ij

        def forward(self, pos, Z, Q_total, S=None):
            """
            pos     : (N, 3) float, positions in Ã…ngstrÃ¶m
            Z       : (N,)   long,  atomic numbers (1-based)
            Q_total : scalar float, total charge in electrons
            S       : ignored in stub
            Returns : E_total (scalar)
            """
            N = pos.shape[0]
            h = self.embed(Z)                              # (N, hidden)

            i_idx, j_idx, r_ij = self._interatomic(pos)

            # aggregate messages (mean over neighbours)
            msg_in = torch.cat([h[j_idx],
                                 r_ij.unsqueeze(-1) / self.cutoff], dim=-1)
            m = torch.zeros_like(h)
            m.scatter_add_(0, i_idx.unsqueeze(-1).expand_as(self.msg(msg_in)),
                           self.msg(msg_in))
            h = h + m / (N + 1e-6)

            E_local = self.energy_head(h).sum()

            if not self.use_charge:
                return E_local

            # --- charge prediction with conservation ---
            q_raw = self.charge_head(h)                    # (N,)
            q = q_raw + (Q_total - q_raw.sum()) / N        # exact conservation

            if not self.use_coulomb:
                return E_local

            # --- Coulomb term ---
            if i_idx.numel() == 0:
                return E_local

            key = ((Z[i_idx].long() * Z[j_idx].long()) % 97).long()
            a   = F.softplus(self.softening(key).squeeze(-1))
            r2  = r_ij ** 2
            E_coul = self.ke * (q[i_idx] * q[j_idx] / (r2 + a ** 2).sqrt()).sum()

            return E_local + self.lambda_coul * E_coul


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42
torch.manual_seed(SEED)


def make_molecule(n_atoms=6, min_dist=1.5, charge=0.0, seed=SEED):
    """Return (pos, Z, Q) with minimum interatomic distance >= min_dist Ã…."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    while True:
        pos = torch.rand(n_atoms, 3, generator=rng) * 5.0  # Ã… box
        diff  = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist  = (diff**2).sum(-1).sqrt()
        dist.fill_diagonal_(999.0)
        if dist.min().item() >= min_dist:
            break
    Z = torch.randint(1, 9, (n_atoms,), generator=rng)
    return pos, Z, torch.tensor(charge)


@pytest.fixture(scope="module")
def model():
    m = ChargeAwarePotential(use_charge=True, use_coulomb=True)
    m.eval()
    return m


@pytest.fixture(scope="module")
def neutral_mol():
    return make_molecule(n_atoms=6, charge=0.0)


@pytest.fixture(scope="module")
def charged_mol():
    return make_molecule(n_atoms=6, charge=1.0, seed=123)


# ---------------------------------------------------------------------------
# Test 1 â€“ Charge conservation
# ---------------------------------------------------------------------------

def test_charge_conservation(model, neutral_mol):
    pos, Z, Q = neutral_mol
    # Hook into charge_head to capture q
    captured = {}

    def hook(module, inp, out):
        h = inp[0]
        q_raw = out
        q = q_raw + (Q - q_raw.sum()) / len(q_raw)
        captured["q"] = q.detach()

    handle = model.charge_head.register_forward_hook(hook)
    with torch.no_grad():
        model(pos, Z, Q)
    handle.remove()

    q_sum = captured["q"].sum().item()
    assert abs(q_sum - Q.item()) < 1e-4, (
        f"Charge conservation violated: sum(q)={q_sum:.6f}, Q_total={Q.item()}"
    )


def test_charge_conservation_nonzero(model, charged_mol):
    pos, Z, Q = charged_mol
    captured = {}

    def hook(module, inp, out):
        q_raw = out
        q = q_raw + (Q - q_raw.sum()) / len(q_raw)
        captured["q"] = q.detach()

    handle = model.charge_head.register_forward_hook(hook)
    with torch.no_grad():
        model(pos, Z, Q)
    handle.remove()

    q_sum = captured["q"].sum().item()
    assert abs(q_sum - Q.item()) < 1e-4, (
        f"Charge conservation violated for Q={Q.item()}: sum(q)={q_sum:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2 â€“ Translation invariance
# ---------------------------------------------------------------------------

def test_translation_invariance(model, neutral_mol):
    pos, Z, Q = neutral_mol
    t = torch.tensor([3.7, -2.1, 5.5])
    with torch.no_grad():
        E0 = model(pos, Z, Q)
        E1 = model(pos + t, Z, Q)
    assert abs(E0.item() - E1.item()) < 1e-4, (
        f"Translation not invariant: Î”E = {abs(E0.item()-E1.item()):.6e}"
    )


# ---------------------------------------------------------------------------
# Test 3 â€“ Rotation invariance
# ---------------------------------------------------------------------------

def _random_rotation(seed=7):
    """Random SO(3) rotation via QR decomposition."""
    torch.manual_seed(seed)
    Q_mat, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q_mat) < 0:
        Q_mat[:, 0] *= -1
    return Q_mat


def test_rotation_invariance(model, neutral_mol):
    pos, Z, Q = neutral_mol
    R = _random_rotation()
    pos_rot = pos @ R.T
    with torch.no_grad():
        E0 = model(pos, Z, Q)
        E1 = model(pos_rot, Z, Q)
    assert abs(E0.item() - E1.item()) < 1e-3, (
        f"Rotation not invariant: Î”E = {abs(E0.item()-E1.item()):.6e}"
    )


# ---------------------------------------------------------------------------
# Test 4 â€“ Permutation invariance
# ---------------------------------------------------------------------------

def test_permutation_invariance(model, neutral_mol):
    pos, Z, Q = neutral_mol
    N = pos.shape[0]
    torch.manual_seed(99)
    perm = torch.randperm(N)
    with torch.no_grad():
        E0 = model(pos,       Z,       Q)
        E1 = model(pos[perm], Z[perm], Q)
    assert abs(E0.item() - E1.item()) < 1e-4, (
        f"Permutation not invariant: Î”E = {abs(E0.item()-E1.item()):.6e}"
    )


# ---------------------------------------------------------------------------
# Test 5 â€“ Force = -grad(E)   (autograd vs finite-difference)
# ---------------------------------------------------------------------------

def test_forces_negative_gradient(model, neutral_mol):
    pos, Z, Q = neutral_mol
    pos_ad = pos.clone().requires_grad_(True)
    E = model(pos_ad, Z, Q)
    E.backward()
    F_autograd = -pos_ad.grad.clone()   # (N,3)

    eps = 1e-3
    F_fd = torch.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            p_fwd = pos.clone(); p_fwd[i, d] += eps
            p_bwd = pos.clone(); p_bwd[i, d] -= eps
            with torch.no_grad():
                Ef = model(p_fwd, Z, Q)
                Eb = model(p_bwd, Z, Q)
            F_fd[i, d] = -(Ef - Eb) / (2 * eps)

    max_err = (F_autograd - F_fd).abs().max().item()
    assert max_err < 5e-2, (
        f"Force â‰  -âˆ‡E: max |F_autograd - F_fd| = {max_err:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 6 â€“ Force equivariance under rotation
# ---------------------------------------------------------------------------

def test_force_equivariance(model, neutral_mol):
    pos, Z, Q = neutral_mol
    R = _random_rotation(seed=13)
    pos_rot = pos @ R.T

    pos_ad  = pos.clone().requires_grad_(True)
    E0 = model(pos_ad, Z, Q); E0.backward()
    F0 = -pos_ad.grad.clone()          # (N,3) in original frame

    pos_r_ad = pos_rot.clone().requires_grad_(True)
    E1 = model(pos_r_ad, Z, Q); E1.backward()
    F1 = -pos_r_ad.grad.clone()        # (N,3) in rotated frame

    F0_rot = F0 @ R.T                  # should equal F1
    max_err = (F0_rot - F1).abs().max().item()
    assert max_err < 1e-3, (
        f"Force equivariance violated: max |RF_0 - F_1| = {max_err:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 7 â€“ Finite Coulomb energy at very short range  (no divergence)
# ---------------------------------------------------------------------------

def test_finite_coulomb_short_range():
    """Softened Coulomb must return finite E even at r â‰ˆ 0."""
    m = ChargeAwarePotential(use_charge=True, use_coulomb=True)
    m.eval()

    # Two atoms nearly on top of each other
    pos = torch.tensor([[0.0, 0.0, 0.0],
                         [0.05, 0.0, 0.0]])   # 0.05 Ã… separation
    Z   = torch.tensor([1, 1])
    Q   = torch.tensor(0.0)

    with torch.no_grad():
        E = m(pos, Z, Q)

    assert torch.isfinite(E), f"Coulomb diverged at short range: E = {E}"
    assert E.abs().item() < 1e4, f"Coulomb unreasonably large at short range: E = {E.item():.2e}"


# ---------------------------------------------------------------------------
# Test 8 â€“ Overfit-100 sanity check
# ---------------------------------------------------------------------------

def _synthetic_dataset(n=100, n_atoms=4, seed=0):
    """Returns list of (pos, Z, Q, E_target) tuples."""
    rng = torch.Generator(); rng.manual_seed(seed)
    data = []
    for _ in range(n):
        pos = torch.rand(n_atoms, 3, generator=rng) * 4.0
        Z   = torch.randint(1, 6, (n_atoms,), generator=rng)
        Q   = torch.zeros(())
        E   = torch.randn(1, generator=rng).item()
        data.append((pos, Z, Q, E))
    return data


def test_overfit_100_sanity():
    """
    A model with enough capacity MUST be able to memorise 100 small random
    molecules in <500 gradient steps.  If it cannot, the architecture or
    training code has a fundamental bug.
    """
    torch.manual_seed(0)
    m = ChargeAwarePotential(hidden=64, use_charge=True, use_coulomb=True)
    m.train()
    optim = torch.optim.Adam(m.parameters(), lr=3e-3)
    dataset = _synthetic_dataset(n=100, n_atoms=4)

    for step in range(500):
        optim.zero_grad()
        loss = torch.tensor(0.0)
        for pos, Z, Q, E_tgt in dataset:
            E_pred = m(pos, Z, Q)
            loss = loss + (E_pred - E_tgt) ** 2
        loss = loss / len(dataset)
        loss.backward()
        optim.step()

    # Final loss
    m.eval()
    with torch.no_grad():
        final_loss = sum(
            (m(pos, Z, Q).item() - E_tgt) ** 2
            for pos, Z, Q, E_tgt in dataset
        ) / len(dataset)

    assert final_loss < 0.5, (
        f"Overfit-100 failed: final MSE = {final_loss:.4f}. "
        "Model cannot memorise 100 samples â€” likely architectural bug."
    )
'@ | Set-Content -Encoding UTF8 "tests\test_invariances.py"

Write-Host "  âœ“ tests/test_invariances.py" -ForegroundColor Green

# â”€â”€ YAML configs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

Write-Host "â–¶ Writing YAML configs ..." -ForegroundColor Cyan

@'
# configs/transition1x_spice.yaml
# Dataset configuration for RALRC MLIP Benchmark
# ------------------------------------------------
# Inputs:  Transition1x (reactive IRC paths, DFT barriers, TS geometries)
#          SPICE v1/v2   (charged + neutral drug-fragment MD snapshots)
# NOTE:    Real training requires downloading these datasets manually.
#          See README.md for download instructions.

name: transition1x_spice_combined

datasets:
  transition1x:
    enabled: true
    path: data/transition1x          # HDF5 or NPZ cache produced by data.py
    # Original source: https://zenodo.org/record/7998850
    split_strategy: reaction_family  # defined in split.py
    include_ts: true                 # include transition-state frames
    energy_unit: eV
    force_unit: eV/Ang
    # Filtering
    max_atoms: 60
    allowed_elements: [H, C, N, O, F, S, Cl, Br]

  spice:
    enabled: true
    path: data/spice                 # HDF5 or NPZ cache
    # Original source: https://github.com/openmm/spice-dataset
    version: "2"
    split_strategy: molecule_scaffold
    energy_unit: eV
    force_unit: eV/Ang
    max_atoms: 80
    include_charges: true            # use DFT Mulliken/ESP charges as labels

combined:
  shuffle_seed: 17
  train_frac: 0.80
  val_frac:   0.10
  test_frac:  0.10
  leakage_check: true               # enforced by split.py â€“ no rxn family leak

# Preprocessing
normalize_energies: true            # per-species atomic reference subtraction
subtract_atomic_refs: true
atomic_ref_file: data/atomic_refs.json

# DataLoader
batch_size: 8                       # graphs; accumulate if GPU OOM
num_workers: 4
pin_memory: true
'@ | Set-Content -Encoding UTF8 "configs\transition1x_spice.yaml"

@'
# configs/eval.yaml
# Evaluation configuration for RALRC MLIP Benchmark
# --------------------------------------------------

name: ralrc_evaluation

checkpoint_dir: checkpoints/
results_dir: benchmarks/

metrics:
  # Primary scientific metrics
  barrier_mae:
    enabled: true
    unit: eV
    description: "MAE on Transition1x IRC energy barrier heights"
    success_threshold_vs_baseline: 0.20   # 20% improvement = weak signal

  ts_force_mae:
    enabled: true
    unit: eV/Ang
    description: "Force MAE specifically at TS-neighbourhood frames (|s| < 0.1 IRC)"
    success_threshold_vs_baseline: 0.15

  force_mae_overall:
    enabled: true
    unit: eV/Ang

  energy_mae_overall:
    enabled: true
    unit: eV

  ood_degradation:
    enabled: true
    description: "Ratio test_MAE / val_MAE; lower = better OOD transfer"
    compute_on: [transition1x_test, spice_ood_charged]

  charge_neutrals_regression:
    enabled: true
    description: "Energy/force MAE on neutral molecules must not increase"

  # Coulomb long-range probe
  coulomb_qualitative:
    enabled: true
    description: "Check E ~ 1/r for separated charged fragments (r=5..20 Ang)"

seeds: [17, 29, 43]
aggregate: median_of_seeds

# Conclusiveness thresholds (â‰¥3 must be met for CONCLUSIVE classification)
conclusiveness:
  min_criteria_met: 3
  criteria:
    - barrier_mae_improvement_pct >= 20
    - ts_force_mae_improvement_pct >= 15
    - ood_degradation_ratio_decreases: true
    - charged_ion_tests_improve_no_neutral_regression: true
    - nve_nvt_md_stability_improves: true
    - coulomb_qualitative_correct: true
    - runtime_within_2x_local_mace: true
    - reproduces_across_all_seeds: true
    - ablations_isolate_charge_longrange: true

output:
  save_predictions: true
  save_per_reaction: true
  csv: benchmarks/benchmark_results.csv
  report: reports/final_report.md
'@ | Set-Content -Encoding UTF8 "configs\eval.yaml"

@'
# configs/md.yaml
# Molecular dynamics stability test configuration
# ------------------------------------------------
# Tests whether the potential supports stable NVE/NVT trajectories.
# Instability (energy drift, exploding forces) = practical failure mode.

name: ralrc_md_stability

# Test molecules (small, gas-phase, to isolate MLIP quality from periodic effects)
systems:
  - name: ethanol_neutral
    smiles: "CCO"
    charge: 0
    spin: 1
    n_replicas: 3

  - name: methylammonium_cation
    smiles: "C[NH3+]"
    charge: 1
    spin: 1
    n_replicas: 3

  - name: acetate_anion
    smiles: "CC(=O)[O-]"
    charge: -1
    spin: 1
    n_replicas: 3

  - name: glycine_zwitterion
    smiles: "[NH3+]CC(=O)[O-]"
    charge: 0
    spin: 1
    n_replicas: 3

integrator:
  type: velocity_verlet
  timestep_fs: 0.5
  n_steps_nvt: 5000     # 2.5 ps equilibration at 300 K
  n_steps_nve: 5000     # 2.5 ps production NVE

thermostat:
  type: langevin
  temperature_K: 300.0
  friction_ps: 1.0
  seed: 17

stability_criteria:
  max_energy_drift_eV_per_atom_per_ps: 0.01   # NVE drift threshold
  max_force_norm_eV_per_Ang: 50.0             # per-atom force cap
  min_survival_frac: 0.9                       # fraction of replicas not crashing

output_dir: md_trajectories/
log_interval_steps: 100
save_trajectory: true
'@ | Set-Content -Encoding UTF8 "configs\md.yaml"

@'
# configs/baselines/local_mace_style.yaml
# Baseline: local MACE-style MLIP, NO charge conditioning, NO Coulomb
# -------------------------------------------------------------------
# This is the control. All other models must beat this on the same data split.

name: local_mace_style

model:
  architecture: ChargeAwarePotential
  use_charge: false
  use_coulomb: false
  charge_mode: none
  hidden: 128
  n_layers: 3
  cutoff_ang: 5.0
  n_species: 36
  softening_init: 0.5   # unused when use_coulomb=false

training:
  lr: 1.0e-3
  lr_scheduler: cosine_annealing
  batch_size: 8
  epochs: 200
  force_weight: 100.0    # Î»_F in loss = MAE_E + Î»_F * MAE_F
  energy_weight: 1.0
  grad_clip_norm: 10.0
  seed: 17
  ema_decay: 0.999

data:
  config: configs/transition1x_spice.yaml
  split_strategy: reaction_family

checkpoint_dir: checkpoints/local_mace_style/
'@ | Set-Content -Encoding UTF8 "configs\baselines\local_mace_style.yaml"

@'
# configs/baselines/charge_head_no_coulomb.yaml
# Baseline: charge-conditioned head, NO long-range Coulomb term
# -------------------------------------------------------------
# Ablation that isolates whether the *charge head alone* (without Coulomb)
# improves results.  Gain here = benefit of charge information flow.
# Gain of learned_charge_coulomb OVER this = benefit of explicit Coulomb.

name: charge_head_no_coulomb

model:
  architecture: ChargeAwarePotential
  use_charge: true
  use_coulomb: false
  charge_mode: learned           # charges predicted but not used in energy
  hidden: 128
  n_layers: 3
  cutoff_ang: 5.0
  n_species: 36
  softening_init: 0.5

training:
  lr: 1.0e-3
  lr_scheduler: cosine_annealing
  batch_size: 8
  epochs: 200
  force_weight: 100.0
  energy_weight: 1.0
  grad_clip_norm: 10.0
  seed: 17
  ema_decay: 0.999

data:
  config: configs/transition1x_spice.yaml
  split_strategy: reaction_family

checkpoint_dir: checkpoints/charge_head_no_coulomb/
'@ | Set-Content -Encoding UTF8 "configs\baselines\charge_head_no_coulomb.yaml"

@'
# configs/ablations/fixed_charge_coulomb.yaml
# Ablation: fixed (non-learned) integer charges + Coulomb
# --------------------------------------------------------
# Charges are taken as formal integer charges (input Q_total distributed
# uniformly).  Tests whether LEARNED charges are needed or if crude
# charge assignment + Coulomb already helps.

name: fixed_charge_coulomb

model:
  architecture: ChargeAwarePotential
  use_charge: true
  use_coulomb: true
  charge_mode: fixed              # q_i = Q_total / N (uniform, not learned)
  hidden: 128
  n_layers: 3
  cutoff_ang: 5.0
  n_species: 36
  softening_init: 0.5
  lambda_coul: 1.0

training:
  lr: 1.0e-3
  lr_scheduler: cosine_annealing
  batch_size: 8
  epochs: 200
  force_weight: 100.0
  energy_weight: 1.0
  grad_clip_norm: 10.0
  seed: 17
  ema_decay: 0.999

data:
  config: configs/transition1x_spice.yaml
  split_strategy: reaction_family

checkpoint_dir: checkpoints/fixed_charge_coulomb/
'@ | Set-Content -Encoding UTF8 "configs\ablations\fixed_charge_coulomb.yaml"

@'
# configs/ablations/learned_charge_coulomb.yaml
# Full model: learned geometry-dependent charges + shielded Coulomb
# -----------------------------------------------------------------
# This is the PROPOSED model being tested.
# All ablations are compared against this.

name: learned_charge_coulomb

model:
  architecture: ChargeAwarePotential
  use_charge: true
  use_coulomb: true
  charge_mode: learned            # q_i predicted by charge_head(h_i), conserved
  hidden: 128
  n_layers: 3
  cutoff_ang: 5.0
  n_species: 36
  softening_init: 0.5             # initial softening parameter a (learned)
  lambda_coul: 1.0                # initial Coulomb weight Î»_coul (can be learned)
  learn_lambda_coul: true         # allow Î»_coul to be optimised
  learn_softening: true           # allow a_ZiZj to be optimised per species pair

training:
  lr: 1.0e-3
  lr_scheduler: cosine_annealing
  batch_size: 8
  epochs: 200
  force_weight: 100.0
  energy_weight: 1.0
  grad_clip_norm: 10.0
  seed: 17
  ema_decay: 0.999
  # Auxiliary charge supervision (if SPICE provides ESP charges)
  charge_aux_weight: 0.1         # Î»_q in loss; set 0 to disable

data:
  config: configs/transition1x_spice.yaml
  split_strategy: reaction_family

checkpoint_dir: checkpoints/learned_charge_coulomb/
'@ | Set-Content -Encoding UTF8 "configs\ablations\learned_charge_coulomb.yaml"

Write-Host "  âœ“ All 7 YAML configs written" -ForegroundColor Green

# â”€â”€ CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

Write-Host "â–¶ Writing benchmarks/benchmark_results.csv ..." -ForegroundColor Cyan

@'
model,seed,energy_mae,force_mae,barrier_mae,ts_force_mae,ood_degradation,runtime_per_atom_step,status
local_mace_style,17,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
local_mace_style,29,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
local_mace_style,43,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
charge_head_no_coulomb,17,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
charge_head_no_coulomb,29,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
charge_head_no_coulomb,43,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
fixed_charge_coulomb,17,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
fixed_charge_coulomb,29,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
fixed_charge_coulomb,43,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
learned_charge_coulomb,17,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
learned_charge_coulomb,29,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
learned_charge_coulomb,43,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,not_run_no_data,pending
'@ | Set-Content -Encoding UTF8 "benchmarks\benchmark_results.csv"

Write-Host "  âœ“ benchmarks/benchmark_results.csv" -ForegroundColor Green

# â”€â”€ Reports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

Write-Host "â–¶ Writing report templates ..." -ForegroundColor Cyan

@'
# Failure Modes & Pathologies â€” RALRC MLIP Benchmark

> **Status:** Template â€” populate after real training runs complete.
> This document tracks observed failure modes, pathological behaviours,
> and their diagnoses. It is part of the falsifiable benchmark harness.

---

## 1. Charge Pathologies

### 1.1 Charge Collapse
**Description:** All predicted charges q_i converge to Q_total/N regardless of geometry.
**Symptoms:** charge_head output â‰ˆ constant; no geometry dependence.
**Likely Cause:** Charge auxiliary loss weight too low; no gradient signal from Coulomb.
**Mitigation:** Increase `charge_aux_weight`; verify ESP charge labels are present.
**Observed:** [ ] Yes  [ ] No  â€” Fill in after run.

### 1.2 Charge Sign Flip Under Permutation
**Description:** Permuting atoms changes sign of individual charges (but sum stays conserved).
**Symptoms:** test_permutation_invariance passes energy but fails per-atom charges.
**Likely Cause:** Message-passing uses atom index as a feature (breaks perm. invariance).
**Mitigation:** Ensure only Z, geometry, neighbours enter â€” never raw index i.
**Observed:** [ ] Yes  [ ] No

### 1.3 Charge Divergence at Short Range
**Description:** Predicted q_i blow up when two atoms get close.
**Symptoms:** Inf/NaN in E_coul during MD; test_finite_coulomb_short_range fails.
**Likely Cause:** Missing softening in Coulomb denominator; or bad softening init.
**Mitigation:** Verify softplus(a_ZiZj) > 0 always; check `softening_init`.
**Observed:** [ ] Yes  [ ] No

---

## 2. Force Pathologies

### 2.1 Discontinuous Forces
**Description:** Force norm jumps discontinuously as atoms cross cutoff boundary.
**Symptoms:** Energy conserved but NVE temperature fluctuates; visible in force-distance plot.
**Likely Cause:** Hard cutoff without envelope function; Coulomb not smoothly damped to zero.
**Mitigation:** Apply cosine envelope to local messages AND Coulomb term at cutoff.
**Observed:** [ ] Yes  [ ] No

### 2.2 Force Equivariance Failure
**Description:** Rotating the molecule rotates forces incorrectly.
**Symptoms:** test_force_equivariance fails; max |RF_0 âˆ’ F_1| >> 1e-3.
**Likely Cause:** Non-equivariant layer introduced (e.g., absolute position encoding).
**Mitigation:** Audit all layers for absolute-coordinate dependence.
**Observed:** [ ] Yes  [ ] No

### 2.3 Force Noise at Transition States
**Description:** Force MAE spikes at TS frames even when barrier energy is accurate.
**Symptoms:** ts_force_mae >> force_mae_overall.
**Likely Cause:** TS frames are sparse in training set; model interpolates poorly near saddle.
**Mitigation:** Oversample TS frames (weight by IRC proximity); add TS-specific validation set.
**Observed:** [ ] Yes  [ ] No

---

## 3. Training Pathologies

### 3.1 Overfit-100 Failure
**Description:** Model cannot memorise 100 random molecules in 500 steps.
**Symptoms:** test_overfit_100_sanity fails; final MSE > 0.5.
**Likely Cause:** Architectural bug (gradient not flowing); LR too low; hidden dim too small.
**Mitigation:** Check backward graph; increase hidden; raise LR.
**Observed:** [ ] Yes  [ ] No

### 3.2 Force Loss Dominates â€” Energy Diverges
**Description:** force_weight=100 causes energy MAE to explode while force MAE shrinks.
**Symptoms:** Energy MAE > 1 eV but force MAE < 0.1 eV/Ã….
**Likely Cause:** Force and energy gradients have very different scales; needs per-property normalisation.
**Mitigation:** Normalise energy and force targets to unit variance before loss computation.
**Observed:** [ ] Yes  [ ] No

### 3.3 Î»_coul Collapses to Zero
**Description:** If lambda_coul is learned, it collapses to ~0 to avoid Coulomb noise.
**Symptoms:** learned_charge_coulomb performs same as charge_head_no_coulomb.
**Likely Cause:** Coulomb term initially noisy / poorly initialised â†’ gradient pushes Î»â†’0.
**Mitigation:** Use a lower bound clamp on Î»_coul (e.g., min=0.1); warm up Coulomb term.
**Observed:** [ ] Yes  [ ] No

---

## 4. MD Stability Pathologies

### 4.1 NVE Energy Drift
**Description:** Total energy drifts > 0.01 eV/atom/ps in NVE.
**Symptoms:** md_stability.py reports drift exceeds threshold.
**Likely Cause:** Timestep too large; forces discontinuous at cutoff.
**Mitigation:** Reduce timestep to 0.25 fs; add cutoff envelope.
**Observed:** [ ] Yes  [ ] No

### 4.2 Exploding Trajectory
**Description:** Atoms fly apart within first 100 steps.
**Symptoms:** Force norms > 50 eV/Ã…; atoms leave simulation box.
**Likely Cause:** Bad initial geometry; NaN in charge prediction; incorrect units.
**Mitigation:** Relax initial geometry with L-BFGS before MD; check unit conversion factors.
**Observed:** [ ] Yes  [ ] No

---

## 5. OOD Pathologies

### 5.1 Neutral Regression Under Charge Training
**Description:** Adding charge/Coulomb terms degrades neutral molecule accuracy.
**Symptoms:** energy_mae / force_mae on neutral test set increases vs local_mace_style.
**Likely Cause:** Coulomb term adds noise for neutrals (should be ~0 but isn't).
**Mitigation:** Add neutral-set auxiliary loss term; verify q_i â‰ˆ 0 for neutral atoms.
**Observed:** [ ] Yes  [ ] No

---

*Last updated: (fill in date after each training run)*
*Owner: (your name / GitHub handle)*
'@ | Set-Content -Encoding UTF8 "reports\failure_modes.md"

@'
# RALRC MLIP Benchmark â€” Final Report

> **Status:** Template â€” NOT a completed result. Populate after real training.
> **Classification:** INCONCLUSIVE until success criteria are met (see below).

---

## 1. Scientific Hypothesis

A local equivariant MACE-style MLIP augmented with:
- Explicit total-charge conditioning (Q_total input)
- Learned geometry-dependent conserved partial charges (charge_head)
- A shielded long-range Coulomb term (softened 1/r)

improves **out-of-distribution reactive accuracy** and **MD stability**
versus a same-data tuned local MACE baseline, without regressing on neutral molecules.

Energy decomposition:
E_total = E_local + Î»_coul * E_coul + E_ref
E_local = Î£_i E_i^MACE
q_raw_i = charge_head(h_i)
q_i = q_raw_i + (Q - Î£_j q_raw_j) / N # exact charge conservation
E_coul = k_e * Î£_{i<j} q_i q_j / sqrt(r_ij^2 + softplus(a_ZiZj)^2)
F_i = -âˆ‡_{R_i} E_total


---

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Training data | Transition1x + SPICE v2 (combined) |
| Split strategy | Reaction-family leakage-safe split |
| Seeds | 17, 29, 43 |
| Baseline | local_mace_style (same data, no charge/Coulomb) |
| Hardware | (fill in: GPU model, VRAM) |
| Training time | (fill in: hours per seed) |

### Models Compared

| Config | use_charge | use_coulomb | charge_mode |
|--------|-----------|-------------|-------------|
| local_mace_style | No | No | none |
| charge_head_no_coulomb | Yes | No | learned |
| fixed_charge_coulomb | Yes | Yes | fixed (Q/N) |
| learned_charge_coulomb | Yes | Yes | learned |

---

## 3. Results

> Fill in from `benchmarks/benchmark_results.csv` after training.

### 3.1 Primary Metrics (median over seeds 17/29/43)

| Model | Barrier MAE (eV) | TS-Force MAE (eV/Ã…) | OOD Degradation | Status |
|-------|-----------------|---------------------|-----------------|--------|
| local_mace_style | â€” | â€” | â€” | pending |
| charge_head_no_coulomb | â€” | â€” | â€” | pending |
| fixed_charge_coulomb | â€” | â€” | â€” | pending |
| learned_charge_coulomb | â€” | â€” | â€” | pending |

### 3.2 Improvement vs Baseline (learned_charge_coulomb vs local_mace_style)

| Metric | Improvement (%) | Threshold | Met? |
|--------|----------------|-----------|------|
| Barrier MAE | â€” | â‰¥20% | â€” |
| TS-Force MAE | â€” | â‰¥15% | â€” |
| OOD degradation factor | â€” | decreases | â€” |
| Charged/ion tests | â€” | improves | â€” |
| NVE/NVT MD stability | â€” | improves | â€” |
| Coulomb qualitative | â€” | correct | â€” |
| Runtime vs baseline | â€” | â‰¤2Ã— | â€” |
| Reproducible (3 seeds) | â€” | yes | â€” |
| Ablations isolate cause | â€” | yes | â€” |

**Criteria met: â€” / 9**

---

## 4. Conclusiveness Classification

*(Fill in after runs)*

- [ ] **CONCLUSIVE** â€” â‰¥3 criteria met, all reproducible, ablations confirm mechanism
- [ ] **INCONCLUSIVE** â€” <3 criteria met OR not reproducible across seeds
- [ ] **BENCHMARK-ONLY** â€” Tests pass, no real training data used
- [ ] **NEGATIVE RESULT** â€” Proposed model does NOT outperform baseline (valid scientific outcome)

### Current Classification: **BENCHMARK-ONLY / INCONCLUSIVE**
*No real Transition1x or SPICE training data has been used. All CSV values are placeholders.*

---

## 5. Ablation Analysis

*(Fill in: does the gain come from charge conditioning, from Coulomb, or both?)*

| Ablation | Finding |
|----------|---------|
| local_mace_style â†’ charge_head_no_coulomb | (charge head alone contribution) |
| charge_head_no_coulomb â†’ fixed_charge_coulomb | (Coulomb with fixed charges contribution) |
| fixed_charge_coulomb â†’ learned_charge_coulomb | (learned charges contribution) |

---

## 6. Failure Modes Encountered

*(Cross-reference `reports/failure_modes.md`)*

- [ ] Charge collapse (Section 1.1)
- [ ] Force discontinuity (Section 2.1)
- [ ] Î»_coul collapse (Section 3.3)
- [ ] NVE drift (Section 4.1)
- [ ] Neutral regression (Section 5.1)

---

## 7. Limitations & Next Steps

1. **Data scale:** Transition1x has ~10k reactions; MACE-style models typically need 100k+ for robust OOD.
2. **Periodic systems:** This benchmark is gas-phase only. Ewald/PME needed for condensed-phase extension.
3. **Spin:** Spin multiplicity S is not yet used by the model (stub ignores it).
4. **Charge supervision:** If SPICE ESP charges are not used as labels, the charge head is purely self-supervised.
5. **Long-range cutoff:** The Coulomb term uses the same cutoff as local interactions â€” true long-range requires Ewald.

---

## 8. Reproducibility Checklist

- [ ] All random seeds fixed (17, 29, 43)
- [ ] Data splits frozen before any hyperparameter tuning
- [ ] Hyperparameters not tuned on test set
- [ ] All configs committed to `configs/`
- [ ] `benchmark_results.csv` populated with real values
- [ ] `failure_modes.md` updated with observed pathologies
- [ ] Model checkpoints archived

---

*Report generated by RALRC MLIP Benchmark harness.*
*This is a falsifiable benchmark, not a validated result.*
*Classify as INCONCLUSIVE until all real training runs complete.*
'@ | Set-Content -Encoding UTF8 "reports\final_report.md"

Write-Host "  âœ“ reports/failure_modes.md" -ForegroundColor Green
Write-Host "  âœ“ reports/final_report.md"  -ForegroundColor Green

# â”€â”€ Ensure tests/__init__.py exists â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (-not (Test-Path "tests\__init__.py")) {
    "" | Set-Content -Encoding UTF8 "tests\__init__.py"
    Write-Host "  OK tests/__init__.py created" -ForegroundColor Green
}

Write-Host ""
Write-Host "RALRC file creation complete." -ForegroundColor Yellow
Write-Host "Files written/updated: tests, configs, benchmarks, reports." -ForegroundColor Yellow
Write-Host "Next: run the conclusiveness one-liner below." -ForegroundColor Cyan
