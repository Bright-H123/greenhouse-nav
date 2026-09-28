"""Local controllers that follow a global path.

PurePursuit is the plain geometric follower used in the first demo.
RegulatedPurePursuit is the baseline in the evaluation. It follows Macenski et al.
(2023), the default path follower in ROS 2 Nav2: adaptive lookahead, speed
regulated by path curvature and by obstacle proximity, rotate-in-place for large
heading errors, and a stop when the commanded arc would collide within 1 s.

The MPC and TD3 controllers live in their own modules and share this interface:

    reset(sim)                      called at episode start
    act(obs, path) -> (v, omega)    one command per 0.1 s step
"""

from __future__ import annotations

import numpy as np

from nav.sim import DT, LIDAR_RANGE, ROBOT_R, V_MAX, W_MAX


def lookahead_point(pos: np.ndarray, path: list[np.ndarray], L: float) -> np.ndarray:
    if not path:
        return pos
    pts = np.asarray(path)
    k = int(np.argmin(np.linalg.norm(pts - pos, axis=1)))
    acc = 0.0
    for i in range(k, len(pts) - 1):
        seg = np.linalg.norm(pts[i + 1] - pts[i])
        if acc + seg >= L:
            return pts[i] + (pts[i + 1] - pts[i]) * (L - acc) / max(seg, 1e-9)
        acc += seg
    return pts[-1]


class PurePursuit:
    name = "D* + pure pursuit"

    def __init__(self, lookahead: float = 1.0, v: float = 0.8):
        self.L, self.v = lookahead, v

    def reset(self, sim) -> None:
        pass

    def act(self, obs: dict, path: list[np.ndarray]):
        x, y, th = obs["state"]
        tgt = lookahead_point(np.array([x, y]), path, self.L)
        d = tgt - np.array([x, y])
        alpha = np.arctan2(d[1], d[0]) - th
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        ld = max(np.linalg.norm(d), 1e-3)
        curv = 2 * np.sin(alpha) / ld
        v = self.v * max(0.25, np.cos(alpha))        # slow down for sharp turns
        if abs(alpha) > np.pi / 2:                    # target behind: turn on the spot
            v = 0.0
        w = np.clip(v * curv if v > 0 else np.sign(alpha) * W_MAX, -W_MAX, W_MAX)
        return float(np.clip(v, 0, V_MAX)), float(w)


class RegulatedPurePursuit:
    """Regulated Pure Pursuit (Macenski et al., 2023) with Nav2's default parameters."""

    name = "D* + RPP"

    def __init__(self, v: float = 0.8, lookahead_time: float = 1.5, l_min: float = 0.3,
                 l_max: float = 0.9, r_min: float = 0.9, v_min_frac: float = 0.25,
                 prox_dist: float = 0.6, rotate_angle: float = 0.785, collision_time: float = 1.0):
        self.v_des, self.t_look, self.l_min, self.l_max = v, lookahead_time, l_min, l_max
        self.r_min, self.v_min = r_min, v_min_frac * v
        self.prox, self.rot, self.t_col = prox_dist, rotate_angle, collision_time

    def reset(self, sim) -> None:
        self.stops = 0

    def _hits(self, obs):
        x, y, th = obs["state"]
        d, rel = obs["lidar"], obs["lidar_rel"]
        m = d < LIDAR_RANGE - 1e-6
        a = th + rel[m]
        return np.stack([x + d[m] * np.cos(a), y + d[m] * np.sin(a)], 1)

    def _arc_collides(self, state, v, w, hits, horizon) -> bool:
        if len(hits) == 0:
            return False
        x, y, th = state
        for _ in range(int(np.ceil(horizon / DT))):
            x += v * np.cos(th) * DT
            y += v * np.sin(th) * DT
            th += w * DT
            if np.min(np.hypot(hits[:, 0] - x, hits[:, 1] - y)) < ROBOT_R:
                return True
        return False

    def act(self, obs: dict, path: list[np.ndarray]):
        if not path:
            return 0.0, 0.0
        state = obs["state"]
        x, y, th = state
        v_now = obs["vel"][0]
        L = float(np.clip(abs(v_now) * self.t_look, self.l_min, self.l_max))
        tgt = lookahead_point(np.array([x, y]), path, L)
        d = tgt - np.array([x, y])
        ld = max(np.linalg.norm(d), 1e-3)
        alpha = (np.arctan2(d[1], d[0]) - th + np.pi) % (2 * np.pi) - np.pi
        if abs(alpha) > self.rot:                       # rotate to heading
            return 0.0, float(np.sign(alpha) * W_MAX)
        curv = 2 * np.sin(alpha) / ld
        v = self.v_des
        if abs(curv) > 1e-6 and 1 / abs(curv) < self.r_min:          # curvature regulation
            v *= (1 / abs(curv)) / self.r_min
        clear = float(np.min(obs["lidar"])) - ROBOT_R                 # proximity regulation
        if clear < self.prox:
            v *= max(clear, 0.0) / self.prox
        v = float(np.clip(max(v, self.v_min), 0.0, V_MAX))
        w = float(np.clip(v * curv, -W_MAX, W_MAX))
        horizon = min(self.t_col, ld / max(v, 1e-3))
        if self._arc_collides(state, v, w, self._hits(obs), horizon):
            self.stops += 1
            return 0.0, 0.0
        return v, w
