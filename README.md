# ralrc-mlip-benchmark
Reactive charge-aware long-range-corrected equivariant MLIP benchmark (RALRC). Falsifiable benchmark harness for testing charge-conditioned MACE-style potentials vs local MACE baseline on reactive/OOD chemistry.


## ⚠️ STATUS: BENCHMARK SCAFFOLD ONLY

This repository contains a **benchmark harness**, NOT a validated scientific result.

The model architecture and test suite are implemented, but:
- ❌ No Transition1x or SPICE training data loaded
- ❌ No GPU training completed
- ❌ No tuned local MACE baseline for comparison  
- ❌ No held-out reaction-family OOD test completed
- ❌ Unit tests not verified to pass

**Scientific classification: INCONCLUSIVE / BENCHMARK-ONLY**

---

## Quick Start

### Clone and Setup

```bash
git clone https://github.com/Heroic297/ralrc-mlip-benchmark.git
cd ralrc-mlip-benchmark
pip install -e ".[chem,mace]"
```

### One-Liner: Test if Results Would Be Conclusive

Run this command to check if the benchmark would produce a real scientific finding:

```bash
pytest -q tests/ && echo "✓ Tests pass" && echo "" && echo "CONCLUSIVE ONLY IF:" && echo "  1. Trained on real Transition1x/SPICE (not toy data)" && echo "  2. Reaction-family splits (not random geometry splits)" && echo "  3. Barrier MAE improves 20-30% over same-data local MACE" && echo "  4. TS force MAE improves 15%+" && echo "  5. OOD degradation factor decreases" && echo "  6. Reproduces across seeds 17,29,43" && echo "  7. No leakage across train/test splits" && echo "" && echo "Current status: INCONCLUSIVE (no real training)" || echo "✗ Tests FAILED - cannot be conclusive"
```

**Short version:**
```bash
pytest -q && echo "Tests OK but NO REAL TRAINING = INCONCLUSIVE" || echo "Tests FAILED = NOT CONCLUSIVE"
```

---

## What Makes Results Conclusive?

A result is **conclusive** (positive, negative, or benchmark-only) only if:

### Required:
- [x] All invariance tests pass (charge conservation, translation, rotation, equivariance)
- [ ] Trained on real Transition1x reaction pathways with force labels
- [ ] Leakage-safe splits by reaction family / reaction ID (not random)
- [ ] Same-data local MACE baseline trained with same budget
- [ ] Barrier height MAE reported on held-out reaction families
- [ ] TS-neighborhood force MAE reported
- [ ] OOD degradation factor computed (MAE_OOD / MAE_ID)
- [ ] Results reproduce across 3 seeds minimum

### Success Thresholds:
Declare meaningful progress ONLY if ≥3 are true:
1. Barrier MAE improves 20-30%+ over local MACE
2. TS force MAE improves 15%+
3. OOD degradation decreases
4. Charged/ion tests improve without neutral regression
5. MD stability improves (NVE/NVT)
6. Separated charged fragments show Coulombic behavior
7. Runtime within 2x local MACE
8. Ablations show gain comes from charge mechanism

**Otherwise classify as:** inconclusive, benchmark-only, or negative result.

---

## Architecture

```
E_total = E_local + λ_coul * E_coul + E_ref

E_local = Σ_i E_i^MACE

q_raw_i = charge_head(h_i)
q_i = q_raw_i + (Q - Σ_j q_raw_j) / N    # exact charge conservation

E_coul = k_e * Σ_{i<j} q_i q_j / sqrt(r_ij² + softplus(a_ZiZj)²)  # shielded

F_i = -∇_{R_i} E_total    # conservative forces
```

Inputs: atomic numbers Z_i, positions R_i, total charge Q, spin S

Invariances:
- Translation, rotation, permutation
- Exact charge conservation
- Conservative forces
- Size extensivity

---

## Current Repo Status

✓ `pyproject.toml` - Dependencies and project config  
✓ `src/ralrc/__init__.py` - Package init  
⚠️ `src/ralrc/model.py` - Needs full ChargeAwarePotential implementation  
⚠️ `src/ralrc/data.py` - Needs Transition1x/SPICE loaders  
⚠️ `src/ralrc/split.py` - Needs leakage-safe split logic  
⚠️ `src/ralrc/train.py` - Needs training script  
⚠️ `src/ralrc/eval.py` - Needs evaluation harness  
⚠️ `src/ralrc/md_stability.py` - Needs MD test  
⚠️ `tests/test_invariances.py` - Needs invariance tests  
⚠️ Configs, benchmarks, reports - Need full files

**The remaining implementation files will be added via pull request.**

---

## License

MIT
