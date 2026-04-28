# Transition1x Split Audit

## Purpose

This audit checks whether the official Transition1x train/val/test groups are usable as leakage-safe heldout splits for reactive-MLIP benchmarking.

Transition1x contains DFT energies and forces for molecular configurations on and around reaction pathways, making split rigor especially important for evaluating transition-state and reactive-region generalization.

## Results

| Check | train vs val | train vs test | val vs test |
|---|---:|---:|---:|
| Formula overlap | 0 | 0 | 0 |
| Formula Jaccard | 0.00% | 0.00% | 0.00% |
| Reaction-ID overlap | 0 | 0 | 0 |
| Reaction-ID Jaccard | 0.0000% | 0.0000% | 0.0000% |
| Reactant hash overlap | 0 | 0 | 0 |
| Product hash overlap | 0 | 0 | 0 |
| Transition-state hash overlap | 0 | 0 | 0 |

## Split Sizes

| Split | Formulas | Reactions |
|---|---:|---:|
| train | 155 | 9561 |
| val | 8 | 225 |
| test | 8 | 287 |

## Interpretation

The official Transition1x splits are formula-disjoint, reaction-ID-disjoint, and endpoint-hash-disjoint across train/val/test.

This supports using Transition1x as a formula-heldout and reaction-heldout benchmark for MLIP generalization in reactive chemistry.

The reactant endpoint hash union is much smaller than the reaction count, indicating repeated reactant structures or repeated reactant hashes within splits. Therefore, the safe claim is cross-split endpoint-hash disjointness, not global uniqueness of all reactants.

## Next Step

Use this leakage-safe split to evaluate whether model errors concentrate near transition-state-like or high-force regions rather than reporting only global force MAE.
