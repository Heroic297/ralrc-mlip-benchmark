# Research Directions: MLIP Benchmark on Transition1x

> Last updated: 2026-04-28

This document records candidate research questions ranked by novelty,
chemical relevance, and feasibility with local compute.

---

## Background

Transition1x is a DFT (wB97x/6-31G(d)) dataset of ~10 k organic reaction
paths covering reactants, transition states, and products with full NEB-style
trajectories [Schreiner et al. 2022].  It was released specifically to support
better benchmarking of MLIPs on *reactive* chemistry — but the dataset itself
is underexploited beyond "train SchNet/PaiNN, report MAE".

---

## RQ1 — Transition-State Force Accuracy Stratified by Reaction Coordinate

**One-line:** Does MLIP force error spike near the TS, and can we quantify the
"dangerous zone" of the reaction path?

**Chemical motivation:**  Force accuracy near the TS governs whether a model
can be used in NEB/nudged-elastic-band refinement.  A model that is accurate in
the reactant/product basin but fails near the TS is useless for TS search even
if its global MAE looks good.

**Why Transition1x:**  We have full trajectories with endpoint labels, so we
can assign a normalized reaction coordinate ξ ∈ [0,1] to every frame (e.g.,
via RMSD interpolation from reactant) and bucket force MAE by ξ.

**Baseline:**  Global force MAE (what papers currently report).

**Metric:**  Force MAE(ξ) curve; particularly the ratio
`force_MAE(ξ ∈ [0.4,0.6]) / force_MAE(ξ ∈ [0,0.1] ∪ [0.9,1.0])`.

**Minimum viable experiment (local):**
1. Train a small SchNet on Transition1x train split (or load a checkpoint).
2. Assign ξ to every validation frame using RMSD from reactant.
3. Plot force MAE vs ξ bin.
4. Compare endpoint-only training vs full-trajectory training on the same curve.

**Novelty assessment:**  The ξ-stratified error curve does not appear in the
original Transition1x paper or in follow-up MLIP benchmarks surveyed as of
2026.  This would be a concrete, reproducible figure.

---

## RQ2 — Endpoint-Only vs Full-Trajectory Training for TS Prediction

**One-line:**  How much does including NEB-path frames improve TS geometry
and energy prediction versus training only on reactant/product/TS triples?

**Chemical motivation:**  High-quality NEB trajectories are expensive to
generate.  If endpoint-only training is nearly as good for TS prediction,
you need far less DFT data per reaction.

**Why Transition1x:**  The dataset uniquely provides both the full NEB path
*and* labeled endpoint structures for the same reactions.

**Baseline:**  Full-trajectory model (current norm).

**Metric:**  TS energy MAE (eV), TS force MAE (eV/Å), TS RMSD (Å) vs DFT.

**Minimum viable experiment:**
1. Split Transition1x train reactions into two loaders:
   - `endpoint_only`: yield only frames with `endpoint ∈ {reactant, product, TS}`
   - `full_traj`: yield all trajectory frames
2. Train identical model on both.
3. Evaluate *only on TS frames* of the val/test splits.

**Novelty assessment:**  Ablation of data-diet for TS tasks.  Likely "useful
engineering" that produces a publishable table but is not a conceptual breakthrough.

---

## RQ3 — Reaction-Family Generalization and OOD Detection

**One-line:**  Can we identify which molecular formulas / reaction families
are systematically harder, and does an uncertainty estimate predict failures?

**Chemical motivation:**  MLIP generalization across chemical space is the
crux of using these models in real reaction discovery workflows.  Knowing
a priori which reactions will fail is actionable.

**Why Transition1x:**  The dataset spans ~10 k reactions across diverse organic
formulas.  We can treat formula / element composition as a proxy for reaction
family and measure per-family error.

**Baseline:**  Uniform error reported as a single number.

**Metric:**  Per-formula force MAE ranking; fraction of reactions where
error > 2× median ("hard reactions"); AUROC of an uncertainty estimator
(e.g., committee disagreement) for identifying hard reactions.

**Minimum viable experiment:**
1. Run inference on the full val split, grouped by formula.
2. Rank formulas by mean force MAE.
3. Check whether atom count, barrier height, or element composition
   predicts the ranking via linear regression.
4. (Optional) train an ensemble of 3 models, measure committee variance
   as an OOD signal.

**Novelty assessment:**  Per-formula failure analysis on Transition1x has not
been published.  This could be a genuine benchmark contribution if the
correlation findings are non-obvious.

---

## RQ4 — Active Learning Frame Selection for Reactive MLIPs

**One-line:**  Can greedy frame selection (highest uncertainty or highest
force magnitude) match the accuracy of full-trajectory training with <30%
of frames?

**Chemical motivation:**  DFT single-points are the real cost.  If you can
choose which 30% of NEB frames to label, you want a selection criterion.
This is the core of active learning for reaction modeling.

**Why Transition1x:**  The full trajectories simulate "oracle" availability —
we can evaluate selection strategies by withholding labels and measuring
how quickly accuracy converges.

**Baseline:**  Random frame subsampling at the same budget.

**Metric:**  Force MAE on held-out frames vs fraction of frames used
(data-efficiency curve).

**Minimum viable experiment:**
1. For each reaction in train, implement three selectors:
   - `random`: uniform sampling
   - `force_mag`: pick highest `|F|_2` frames (proxy for TS proximity)
   - `farthest_point`: greedy max-min RMSD selection
2. Train on selected subset; evaluate on full val trajectories.
3. Plot MAE vs % frames retained.

**Novelty assessment:**  Frame-selection for NEB-trajectory labeling using
force magnitude or FPS is not well-studied on Transition1x.  This could
produce a data-efficiency curve that is directly useful for future dataset
curation.

---

## RQ5 — Split Quality Audit: Are Train/Val/Test Reactions Truly Independent?

**One-line:**  Do the official splits contain formula or substructure leakage?

**Chemical motivation:**  If the val/test reactions share formulas or
structural motifs with training reactions, reported generalization is
artificially optimistic — a known problem in ML chemistry benchmarks.

**Why Transition1x:**  The split methodology is not documented beyond
"random reaction-level split," leaving open the question of whether
formula identity or atom-count similarity creates implicit leakage.

**Baseline:**  Claimed random split.

**Metric:**  Fraction of val/test formulas seen in train;
Tanimoto similarity of reaction SMILES between splits (if SMILES available);
error stratified by "formula seen in train" vs "unseen formula".

**Minimum viable experiment:**
1. Enumerate all formulas in train vs val vs test.
2. Report overlap fractions.
3. Compare model error on overlapping vs non-overlapping formulas.

**Novelty assessment:**  Benchmark auditing is genuinely useful and publishable
as a short note/letter.  This is lower-effort than the other RQs and could
form a strong methods section for a larger paper.

---

## Recommended Starting Order

| Priority | RQ | Why first |
|---|---|---|
| 1 | RQ5 (split audit) | 0 training needed; results in <1 day |
| 2 | RQ1 (ξ-stratified error) | single training run + analysis |
| 3 | RQ2 (endpoint ablation) | two training runs, clear table |
| 4 | RQ3 (OOD/family) | needs inference at scale |
| 5 | RQ4 (active learning) | most engineering overhead |

---

## Infrastructure Needed

- [x] Lazy HDF5 loader (`src/ralrc/transition1x.py`)
- [x] Dataset summary script (`scripts/dataset_summary.py`)
- [ ] Reaction-coordinate (ξ) assignment utility
- [ ] Per-reaction error aggregation in eval pipeline
- [ ] Minimal SchNet/PaiNN training harness (or load pretrained checkpoint)
- [ ] Formula-level split audit script
