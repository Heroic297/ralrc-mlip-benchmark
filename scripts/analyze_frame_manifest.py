from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MANIFEST = Path("reports/dataset_checks/frame_manifest_pilot.csv")
OUTDIR = Path("reports/figures")
TABDIR = Path("reports/tables")

OUTDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

# Reaction-level table prevents long trajectories from dominating split-level statistics.
rxn = (
    df.groupby(["split", "formula", "rxn_id"], as_index=False)
      .agg(
          n_sampled_frames=("frame_idx", "count"),
          n_frames=("n_frames", "first"),
          n_atoms=("n_atoms", "first"),
          endpoint_barrier_raw=("endpoint_barrier_raw", "first"),
          traj_barrier_raw=("traj_barrier_raw", "first"),
          force_rms_mean=("force_rms", "mean"),
          force_rms_median=("force_rms", "median"),
          force_rms_max=("force_rms", "max"),
          force_l2_mean=("force_l2", "mean"),
          force_l2_max=("force_l2", "max"),
      )
)

split_summary = (
    rxn.groupby("split")
       .agg(
           n_reactions=("rxn_id", "count"),
           n_formulas=("formula", "nunique"),
           n_atoms_median=("n_atoms", "median"),
           n_frames_median=("n_frames", "median"),
           endpoint_barrier_median=("endpoint_barrier_raw", "median"),
           endpoint_barrier_p95=("endpoint_barrier_raw", lambda x: np.percentile(x, 95)),
           force_rms_median=("force_rms_median", "median"),
           force_rms_max_median=("force_rms_max", "median"),
       )
       .reset_index()
)

df["xi_bin"] = pd.cut(
    df["xi_index"],
    bins=np.linspace(0, 1, 11),
    include_lowest=True,
)

xi_force = (
    df.groupby(["split", "xi_bin"], observed=True)
      .agg(
          n_frames=("force_rms", "count"),
          force_rms_mean=("force_rms", "mean"),
          force_rms_median=("force_rms", "median"),
          force_rms_p90=("force_rms", lambda x: np.percentile(x, 90)),
          energy_rel_mean=("energy_rel_first_raw", "mean"),
          energy_rel_median=("energy_rel_first_raw", "median"),
      )
      .reset_index()
)

split_summary.to_csv(TABDIR / "pilot_split_summary_reaction_weighted.csv", index=False)
rxn.to_csv(TABDIR / "pilot_reaction_level_manifest.csv", index=False)
xi_force.to_csv(TABDIR / "pilot_xi_force_bins.csv", index=False)

# Figure 1: sampled rows and reactions by split
fig, ax = plt.subplots(figsize=(7, 4))
counts = df.groupby("split").size().reindex(["train", "val", "test"])
ax.bar(counts.index, counts.values)
ax.set_ylabel("Sampled frames")
ax.set_title("Pilot manifest sampled frames by split")
fig.tight_layout()
fig.savefig(OUTDIR / "pilot_sampled_frames_by_split.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
rcounts = rxn.groupby("split").size().reindex(["train", "val", "test"])
ax.bar(rcounts.index, rcounts.values)
ax.set_ylabel("Reactions")
ax.set_title("Pilot manifest reactions by split")
fig.tight_layout()
fig.savefig(OUTDIR / "pilot_reactions_by_split.png", dpi=200)
plt.close(fig)

# Figure 2: force RMS vs reaction coordinate
fig, ax = plt.subplots(figsize=(8, 5))
for split in ["train", "val", "test"]:
    sub = xi_force[xi_force["split"] == split].copy()
    x = np.arange(len(sub))
    ax.plot(x, sub["force_rms_median"], marker="o", label=f"{split} median")
    ax.fill_between(
        x,
        sub["force_rms_median"],
        sub["force_rms_p90"],
        alpha=0.15,
    )

ax.set_xticks(np.arange(10))
ax.set_xticklabels([f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)], rotation=45)
ax.set_xlabel("Normalized path index bin")
ax.set_ylabel("Force RMS, raw units")
ax.set_title("Force magnitude along reaction path, pilot sample")
ax.legend()
fig.tight_layout()
fig.savefig(OUTDIR / "pilot_force_rms_vs_xi.png", dpi=200)
plt.close(fig)

# Figure 3: endpoint barrier distribution, reaction weighted
fig, ax = plt.subplots(figsize=(8, 5))
for split in ["train", "val", "test"]:
    vals = rxn.loc[rxn["split"] == split, "endpoint_barrier_raw"].dropna()
    ax.hist(vals, bins=25, alpha=0.45, label=split)

ax.set_xlabel("Endpoint barrier, raw dataset units")
ax.set_ylabel("Reaction count")
ax.set_title("Endpoint barrier distribution, pilot reactions")
ax.legend()
fig.tight_layout()
fig.savefig(OUTDIR / "pilot_endpoint_barrier_distribution.png", dpi=200)
plt.close(fig)

# Figure 4: reaction-level max force by barrier
fig, ax = plt.subplots(figsize=(7, 5))
for split in ["train", "val", "test"]:
    sub = rxn[rxn["split"] == split]
    ax.scatter(
        sub["endpoint_barrier_raw"],
        sub["force_rms_max"],
        s=18,
        alpha=0.6,
        label=split,
    )

ax.set_xlabel("Endpoint barrier, raw dataset units")
ax.set_ylabel("Max sampled force RMS, raw units")
ax.set_title("High-force reactions vs endpoint barrier, pilot")
ax.legend()
fig.tight_layout()
fig.savefig(OUTDIR / "pilot_force_vs_barrier.png", dpi=200)
plt.close(fig)

print("Wrote:")
for p in sorted(OUTDIR.glob("pilot_*.png")):
    print(" ", p)
for p in sorted(TABDIR.glob("pilot_*.csv")):
    print(" ", p)

print("\nReaction-weighted split summary:")
print(split_summary.to_string(index=False))

print("\nXi-bin force summary:")
print(xi_force.to_string(index=False))
