"""Train one PPO agent for a given reward config and seed.

Usage:
    python train.py --config baseline --seed 0 --steps 1000000
"""
import argparse
import os

import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import torch

import bipedal_walker_ucd
from rewards import CustomRewardWrapper

# Reward weights per experiment. "task2_original" / "task3_original" are the
# exact weights from the notebook; "rebalanced*" fix the per-step penalty scale
# so that standing still (or falling early) is no longer the best strategy.
CONFIGS = {
    "baseline": {},
    "task2_original": dict(action_penalty_weight=3.0, stability_weight=1.5,
                           smoothness_weight=0.3, speed_weight=0.5, target_speed=1.5),
    "task3_original": dict(action_penalty_weight=2.0, stability_weight=1.0,
                           smoothness_weight=0.1, speed_weight=0.3, target_speed=1.5,
                           bounce_weight=0.4, leg_symmetry_weight=0.5),
    "rebalanced": dict(action_penalty_weight=1.0, stability_weight=0.5,
                       smoothness_weight=0.1),
    "rebalanced_bounce_sym": dict(action_penalty_weight=1.0, stability_weight=0.5,
                                  smoothness_weight=0.1, bounce_weight=0.1,
                                  leg_symmetry_weight=0.1),
}

# Hyper-parameters from the assignment brief (unchanged across all runs).
PPO_KWARGS = dict(batch_size=64, clip_range=0.2, ent_coef=0.001, gae_lambda=0.95,
                  gamma=0.99, learning_rate=0.00025, n_steps=2048, n_epochs=10)


def make_env(config, time_limit=2000):
    env = bipedal_walker_ucd.BipedalWalkerUCD(render_mode=None)
    env = CustomRewardWrapper(env, **CONFIGS[config])
    return Monitor(gym.wrappers.TimeLimit(env, max_episode_steps=time_limit))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=CONFIGS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    args = p.parse_args()

    torch.set_num_threads(1)
    run = f"{args.config}_s{args.seed}"
    agent = sb3.PPO("MlpPolicy", make_env(args.config), seed=args.seed, device="cpu",
                    verbose=0, **PPO_KWARGS)
    agent.set_logger(configure(f"logs/{run}", ["csv"]))  # learning curve -> progress.csv
    agent.learn(total_timesteps=args.steps)
    os.makedirs("agents/runs", exist_ok=True)
    agent.save(f"agents/runs/{run}")
    print(f"saved agents/runs/{run} ({agent.num_timesteps} steps)")


if __name__ == "__main__":
    main()
