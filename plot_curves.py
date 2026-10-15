"""Plot training curves (episode length and base-env progress) from logs/*/progress.csv."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

runs = {}
for f in sorted(glob.glob("logs/*/progress.csv")):
    config = os.path.basename(os.path.dirname(f)).rsplit("_s", 1)[0]
    runs.setdefault(config, []).append(pd.read_csv(f))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for config, dfs in runs.items():
    for ax, col in zip(axes, ["rollout/ep_rew_mean", "rollout/ep_len_mean"]):
        merged = pd.concat([d.set_index("time/total_timesteps")[col] for d in dfs], axis=1)
        mean, std = merged.mean(axis=1), merged.std(axis=1).fillna(0)
        line, = ax.plot(mean.index, mean, label=f"{config} (n={len(dfs)})")
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=line.get_color())
axes[0].set_title("Training reward (each config's own shaped reward)")
axes[1].set_title("Episode length (steps; falls end episodes early)")
for ax in axes:
    ax.set_xlabel("timesteps")
    ax.grid(alpha=0.3)
axes[0].set_yscale("symlog", linthresh=100)
axes[1].legend(fontsize=8)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/training_curves.png", dpi=120)
print("saved results/training_curves.png")
