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
