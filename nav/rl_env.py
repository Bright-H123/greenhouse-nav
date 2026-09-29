"""Gymnasium view of the greenhouse for TD3.

Two variants share one class:

  waypoint mode   the policy is a LOCAL controller under D* Lite. Its target is a
                  point a fixed distance along the current global path, the same
                  lookahead point the pure-pursuit and MPC controllers chase, so
                  the three are compared on equal terms.
  goal mode       end-to-end. No global planner, the target is the final goal.
                  This is the variant that has to discover its own detours.

The policy sees only LiDAR, never worker positions. The classical pipeline does
receive detected worker positions, so the comparison between them is not
like-for-like on inputs; that asymmetry is stated in the README rather than
engineered away, because it is the honest version of "learned policy from raw
sensing" against "modular pipeline with a detector".

Observation (float32):
    [0:N_BINS]      LiDAR min-pooled into N_BINS sectors, divided by LIDAR_RANGE
    [N_BINS]        distance to target, divided by DIST_SCALE, clipped to 1
    [N_BINS+1:+3]   cos and sin of the bearing to the target in the robot frame
    [N_BINS+3]      v / V_MAX
    [N_BINS+4]      omega / W_MAX

Action: (v, omega) as two values in [-1, 1]. v is mapped to [0, V_MAX], so the
robot cannot reverse, matching the other controllers.

The lookahead point steers, the goal pays. Shaping is the reduction in distance
to the FINAL goal in both modes, never to the lookahead point, because the
lookahead point advances along the path as the robot does. Measured under a
working pure-pursuit controller, tracking the lookahead point yields 0.18 m of
credited progress over 300 steps, so the shaping term all but vanishes and only
the penalties remain.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from nav.controllers import lookahead_point
from nav.planner import GridPlanner
from nav.runner import REPLAN_EVERY
from nav.sim import DT, LIDAR_RANGE, ROBOT_R, V_MAX, W_MAX, GreenhouseSim
from nav.world import Greenhouse

N_BINS = 36
DIST_SCALE = 12.0
LOOKAHEAD = 1.0

# Reward. The shaping term is the reduction in distance to the target, which is
# the only term that scales with progress; the rest are bounded penalties.
R_REACH = 60.0
R_COLLIDE = -60.0
W_PROGRESS = 12.0
W_CLEAR = 1.2          # penalty weight when inside CLEAR_AT of an obstacle
CLEAR_AT = 0.55
W_SPIN = 0.05
R_STEP = -0.02         # small per-step cost, so dithering is not free


def bins(lidar: np.ndarray) -> np.ndarray:
    """Min-pool the scan; the nearest return in a sector is what matters."""
    return lidar.reshape(N_BINS, -1).min(1) / LIDAR_RANGE


def make_obs(raw: dict, target: np.ndarray) -> np.ndarray:
    """The policy input. Free of any planner state, so the training env and the
    evaluation adapter cannot drift apart."""
    x, y, th = raw["state"]
    d = np.asarray(target, float) - np.array([x, y])
    dist = float(np.linalg.norm(d))
    bearing = np.arctan2(d[1], d[0]) - th
    v, w = raw["vel"]
    return np.concatenate([
        bins(raw["lidar"]),
        [min(dist / DIST_SCALE, 1.0), np.cos(bearing), np.sin(bearing), v / V_MAX, w / W_MAX],
    ]).astype(np.float32)


def target_point(mode: str, sim, path) -> np.ndarray:
    if mode == "goal" or not path:
        return sim.goal
    return lookahead_point(sim.state[:2], path, LOOKAHEAD)


class GreenhouseRLEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, mode: str = "waypoint", n_workers: int = 4, max_steps: int = 900,
                 house: Greenhouse | None = None, render_mode: str | None = None):
        if mode not in ("waypoint", "goal"):
            raise ValueError(mode)
        self.mode = mode
        self.sim = GreenhouseSim(house or Greenhouse(), n_workers=n_workers, max_steps=max_steps)
        self.planner = GridPlanner(self.sim.house) if mode == "waypoint" else None
        self.render_mode = render_mode
        self.observation_space = spaces.Box(-1.0, 1.0, (N_BINS + 5,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.path: list = []

    # ------------------------------------------------------------------ helpers
    def _target(self) -> np.ndarray:
        return target_point(self.mode, self.sim, self.path)

    def _obs(self, raw: dict) -> np.ndarray:
        return make_obs(raw, self._target())

    def _replan(self, raw: dict) -> None:
        if self.planner is not None and self.sim.t % REPLAN_EVERY == 0:
            self.path = self.planner.replan(raw["state"][:2], raw["workers"])

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        raw = self.sim.reset(seed=seed)
        if self.planner is not None:
            self.planner.reset(raw["state"][:2], self.sim.goal)
            self.path = []
        self._replan(raw)
        self._prev_dist = float(np.linalg.norm(self.sim.goal - self.sim.state[:2]))
        self._raw = raw
        return self._obs(raw), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        v = float((a[0] + 1.0) * 0.5 * V_MAX)
        w = float(a[1] * W_MAX)
        raw, info = self.sim.step((v, w))
        self._replan(raw)

        dist = float(np.linalg.norm(self.sim.goal - self.sim.state[:2]))
        clear = float(np.min(raw["lidar"])) - ROBOT_R
        reward = R_STEP
        reward += W_PROGRESS * (self._prev_dist - dist)
        if clear < CLEAR_AT:
            reward -= W_CLEAR * (CLEAR_AT - max(clear, 0.0)) / CLEAR_AT
        reward -= W_SPIN * abs(raw["vel"][1]) / W_MAX
        if info["reached"]:
            reward += R_REACH
        elif info["collision"]:
            reward += R_COLLIDE
        self._prev_dist = dist
        self._raw = raw

        terminated = bool(info["reached"] or info["collision"])
        truncated = bool(info["timeout"])
        return self._obs(raw), float(reward), terminated, truncated, info

    def render(self):
        from nav.render import draw
        return draw(self.sim, path=self.path, title=f"TD3 ({self.mode})")
