"""Reward-shaping wrapper used in the experiments (copied from the notebook, Task 3 version).

All weights default to 0, which reproduces the base BipedalWalkerUCD reward.
`bounce` and `leg_symmetry` are the two terms I designed; the other four were
provided by the course template.
"""
from collections import deque

import gymnasium as gym
import numpy as np

import bipedal_walker_ucd


class CustomRewardWrapper(gym.Wrapper):
    
    def __init__(self, env, window=20,
                 action_penalty_weight=0.0,
                 stability_weight=0.0,
                 smoothness_weight=0.0,
                 speed_weight=0.0,
                 target_speed=1.5,
                 bounce_weight=0.0,
                 leg_symmetry_weight=0.0):
        super().__init__(env)
        self.action_penalty_weight = action_penalty_weight
        self.stability_weight      = stability_weight
        self.smoothness_weight     = smoothness_weight
        self.speed_weight          = speed_weight
        self.target_speed          = target_speed
        self.bounce_weight         = bounce_weight
        self.leg_symmetry_weight   = leg_symmetry_weight
        self.prev_action  = np.zeros(env.action_space.shape)
        self.hip1_history = deque(maxlen=window)
        self.hip2_history = deque(maxlen=window)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        hull       = self.env.unwrapped.hull
        hull_angle = hull.angle
        hull_vel_x = hull.linearVelocity[0]
        hull_vel_y = hull.linearVelocity[1]   # vertical velocity

        hip1_angle = float(obs[4])   # hip 1 joint angle from observation
        hip2_angle = float(obs[10])  # hip 2 joint angle from observation

        # Standard components
        action_penalty = 0.0
        for a in action:
            action_penalty -= 0.00035 * bipedal_walker_ucd.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
        action_penalty *= self.action_penalty_weight

        stability     = -abs(hull_angle)                            * self.stability_weight
        smoothness    = -np.sum(np.abs(action - self.prev_action))  * self.smoothness_weight
        speed_shaping = -abs(hull_vel_x - self.target_speed)        * self.speed_weight

        # Reward vertical hull movement to encourage a bouncy stride.
        # Squaring means both rising and falling count equally.
        # We clip at 2.0 so a hard landing does not give a huge spike.
        bounce = np.clip(hull_vel_y ** 2, 0.0, 2.0) * self.bounce_weight

        # Penalise both hips swinging the same way at the same time.
        # If the hips are perfectly alternating, hip1 + hip2 cancels to zero.
        # The more they move in sync, the larger the squared penalty becomes.
        leg_symmetry = -(hip1_angle + hip2_angle) ** 2 * self.leg_symmetry_weight

        modified_reward = (base_reward + action_penalty + stability
                           + smoothness + speed_shaping + bounce + leg_symmetry)

        info["reward_components"] = {
            "base": base_reward, "action_penalty": action_penalty,
            "stability": stability, "smoothness": smoothness,
            "speed_shaping": speed_shaping, "bounce": bounce,
            "leg_symmetry": leg_symmetry, "total": modified_reward,
        }
        self.prev_action = action.copy()
        return obs, modified_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.prev_action = np.zeros(self.env.action_space.shape)
        self.hip1_history.clear()
        self.hip2_history.clear()
        return self.env.reset(**kwargs)
