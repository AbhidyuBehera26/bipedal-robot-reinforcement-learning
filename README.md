# Reinforcement Learning Bipedal Walker: PPO Locomotion & Reward Shaping

A machine learning project where I trained a bipedal robot to walk from scratch using PPO (Proximal Policy Optimization) and Stable-Baselines3. I experimented with different reward functions to see how adding penalties for energy and instability could turn a jerky, falling robot into a smooth walker!

## Project Structure
* `bipedal_walker_ucd.py` - Custom Bipedal Walker environment wrapper script.
* `COMP47590 Assignment 2 Going The Distance Template v4-checkpoint.ipynb` - Main Jupyter notebook containing training setups, agent runs, and reflections.
* `agents/` - Saved model checkpoints.
* `eval_videos/` - Rendered evaluation video outputs showing agent progress.

## Setup & Running
1. Install dependencies:
   ```bash
   pip install gymnasium[box2d] stable-baselines3[extra] torch imageio