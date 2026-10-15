"""Evaluate saved PPO agents over many seeded episodes and report outcome metrics.

Metrics per agent (deterministic policy, fixed seeds):
  - success rate: reached the end of the course without falling
  - fall rate: hull touched the ground (episode terminated with -100)
  - distance: % of the course covered (hull x / finish line x)
  - base reward: BipedalWalkerUCD reward (forward progress, -100 on fall)
  - v3 score: the same episode scored with the official BipedalWalker-v3 reward
    (base reward + energy cost + head-angle shaping); 300 = "solved"
  - energy: mean |action| per step (motor effort, 0..1)
  - jerk: mean |a_t - a_{t-1}| per step (lower = smoother)
  - hull tilt: mean |hull angle| in radians (lower = more upright)

Usage:
    python evaluate.py --episodes 50 --max-steps 2000

Every agent is run on the same 50 seeded terrains, with both the deterministic
(mean) action and the stochastic (sampled) action. Results for the runs in
agents/runs/ are aggregated per reward config: mean ± std across training seeds.
"""
import argparse
import glob
import json
import multiprocessing
import os

import numpy as np
import stable_baselines3 as sb3

import bipedal_walker_ucd as bw

FINISH_X = (bw.TERRAIN_LENGTH - bw.TERRAIN_GRASS) * bw.TERRAIN_STEP
START_X = bw.TERRAIN_STEP * bw.TERRAIN_STARTPAD / 2


def run_episode(env, policy, seed, max_steps):
    obs, _ = env.reset(seed=seed)
    prev_action = np.zeros(env.action_space.shape)
    base_reward = 0.0
    energy_cost = 0.0
    abs_actions, jerks, tilts = [], [], []
    fell = finished = False
    steps = 0
    first_angle = last_angle = abs(env.hull.angle)
    for steps in range(1, max_steps + 1):
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        clipped = np.clip(np.abs(action), 0, 1)
        base_reward += reward
        energy_cost += 0.00035 * bw.MOTORS_TORQUE * clipped.sum()
        abs_actions.append(clipped.mean())
        jerks.append(np.abs(action - prev_action).mean())
        tilts.append(abs(env.hull.angle))
        prev_action = action
        if not (env.game_over or env.hull.position[0] < 0):
            last_angle = abs(env.hull.angle)
        if terminated:
            fell = env.game_over or env.hull.position[0] < 0
            finished = not fell
            break
    x = env.hull.position[0]
    # Official v3 reward = UCD reward - energy cost - head-angle shaping.
    # The head shaping is potential-based (shaping_t - shaping_{t-1}) so it
    # telescopes to -5 * (|angle_T| - |angle_0|); on a fall the last step's
    # reward is overwritten with -100, so the sum stops at the step before.
    head_term = -5.0 * (last_angle - first_angle)
    return {
        "success": finished,
        "fell": fell,
        "distance_pct": 100.0 * np.clip((x - START_X) / (FINISH_X - START_X), 0, 1),
        "steps": steps,
        "base_reward": base_reward,
        "v3_score": base_reward - energy_cost + head_term,
        "energy": float(np.mean(abs_actions)),
        "jerk": float(np.mean(jerks)),
        "tilt": float(np.mean(tilts)),
    }


def summarise(results):
    def ms(key):
        v = np.array([r[key] for r in results], dtype=float)
        return float(v.mean()), float(v.std())

    succ = [r for r in results if r["success"]]
    return {
        "episodes": len(results),
        "success_rate": 100.0 * len(succ) / len(results),
        "fall_rate": 100.0 * sum(r["fell"] for r in results) / len(results),
        "distance_pct": ms("distance_pct"),
        "base_reward": ms("base_reward"),
        "v3_score": ms("v3_score"),
        "steps_to_finish": float(np.mean([r["steps"] for r in succ])) if succ else None,
        "energy": ms("energy"),
        "jerk": ms("jerk"),
        "tilt": ms("tilt"),
    }


def evaluate_policy(job):
    """Worker: (label, model path or None, deterministic, episodes, max_steps)."""
    label, path, deterministic, episodes, max_steps = job
    env = bw.BipedalWalkerUCD(render_mode=None)
    if path is None:
        rng = np.random.default_rng(0)
        policy = lambda obs: rng.uniform(-1, 1, size=4)
    else:
        model = sb3.PPO.load(path, device="cpu")
        policy = lambda obs: model.predict(obs, deterministic=deterministic)[0]
    seeds = range(1000, 1000 + episodes)  # same terrains for every agent
    return label, path, deterministic, summarise(
        [run_episode(env, policy, s, max_steps) for s in seeds])


def aggregate(per_seed):
    """Mean and std across training seeds of each per-seed metric."""
    out = {"n_seeds": len(per_seed)}
    for key in ["success_rate", "fall_rate", "distance_pct", "base_reward",
                "v3_score", "energy", "jerk", "tilt"]:
        vals = [r[key][0] if isinstance(r[key], tuple | list) else r[key] for r in per_seed]
        out[key] = (float(np.mean(vals)), float(np.std(vals)))
    out["per_seed_success"] = [r["success_rate"] for r in per_seed]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results/eval_results.json")
    args = p.parse_args()

    jobs = [("random", None, False)]
    for name in ["Basic_Reward", "Task2_Smooth_Gait"]:  # original notebook checkpoints
        for det in (True, False):
            jobs.append((f"notebook/{name}", f"agents/{name}.zip", det))
    for f in sorted(glob.glob("agents/runs/*.zip")):
        config = os.path.basename(f).rsplit("_s", 1)[0]
        for det in (True, False):
            jobs.append((config, f, det))
    jobs = [j + (args.episodes, args.max_steps) for j in jobs]

    with multiprocessing.Pool(args.workers) as pool:
        raw = pool.map(evaluate_policy, jobs)

    grouped = {}
    for label, path, det, summary in raw:
        mode = "deterministic" if det else "stochastic"
        grouped.setdefault(f"{label} [{mode}]", []).append(summary)
    results = {k: aggregate(v) for k, v in grouped.items()}

    for k, r in results.items():
        print(f"{k:45s} seeds={r['n_seeds']} success {r['success_rate'][0]:5.1f}±{r['success_rate'][1]:4.1f}%  "
              f"fall {r['fall_rate'][0]:5.1f}%  dist {r['distance_pct'][0]:5.1f}%  "
              f"base {r['base_reward'][0]:6.1f}  v3 {r['v3_score'][0]:6.1f}  "
              f"energy {r['energy'][0]:.3f}  jerk {r['jerk'][0]:.3f}  tilt {r['tilt'][0]:.3f}  "
              f"per-seed {r['per_seed_success']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
