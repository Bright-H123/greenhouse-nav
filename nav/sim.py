"""2D kinematic greenhouse simulator.

Robot: unicycle model for a skid-steer base of the AgileX Scout Mini's footprint,
approximated by a disc. State (x, y, theta), control (v, omega), 10 Hz.

Workers: discs that walk between random aisle points and pause, sensed by the
robot only through its LiDAR.

LiDAR: 2D ray casting against racks, walls and workers.

Everything is numpy. There is no physics engine; contact is a collision and ends
the episode. That is the right fidelity for comparing planners, and it is what
lets a TD3 policy train on a CPU in about an hour rather than two days in Gazebo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nav.world import Greenhouse, Rect

DT = 0.1
ROBOT_R = 0.33           # Scout Mini is about 0.61 m x 0.58 m
V_MAX, W_MAX = 1.0, 1.5  # m/s, rad/s
A_MAX, ALPHA_MAX = 1.5, 3.0  # per-second limits on the change of v and omega

LIDAR_BEAMS = 36
LIDAR_FOV = np.deg2rad(360.0)
LIDAR_RANGE = 8.0

WORKER_R = 0.30
WORKER_SPEED = (0.6, 1.1)
GOAL_TOL = 0.4


@dataclass
class Worker:
    pos: np.ndarray
    target: np.ndarray
    speed: float
    pause: float = 0.0


def _ray_rect(o: np.ndarray, d: np.ndarray, r: Rect) -> np.ndarray:
    """Distance along rays o + t d to an axis-aligned rectangle (slab method)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tx0 = (r.x0 - o[0]) / d[:, 0]
        tx1 = (r.x1 - o[0]) / d[:, 0]
        ty0 = (r.y0 - o[1]) / d[:, 1]
        ty1 = (r.y1 - o[1]) / d[:, 1]
    tmin = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
    tmax = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
    hit = (tmax >= np.maximum(tmin, 0.0))
    t = np.where(tmin >= 0, tmin, tmax)
    return np.where(hit & (t >= 0), t, np.inf)


def _ray_circle(o: np.ndarray, d: np.ndarray, c: np.ndarray, rad: float) -> np.ndarray:
    oc = o - c
    b = d @ oc
    cc = oc @ oc - rad * rad
    disc = b * b - cc
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0) & (t >= 0), t, np.inf)


class GreenhouseSim:
    def __init__(self, house: Greenhouse | None = None, n_workers: int = 0,
                 max_steps: int = 900, seed: int | None = None):
        self.house = house or Greenhouse()
        self.n_workers = n_workers
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        rel = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, LIDAR_BEAMS, endpoint=False)
        self.beam_rel = rel

    # ---------------------------------------------------------------- episodes
    def reset(self, seed: int | None = None, start=None, goal=None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        margin = ROBOT_R + 0.25
        if start is None or goal is None:
            while True:
                s, g = self.house.aisle_points(self.rng, 2, margin)
                if np.linalg.norm(g - s) > 10.0:          # force non-trivial routes
                    break
            start = s if start is None else np.asarray(start, float)
            goal = g if goal is None else np.asarray(goal, float)
        self.state = np.array([start[0], start[1], self.rng.uniform(-np.pi, np.pi)])
        self.vel = np.zeros(2)
        self.goal = np.asarray(goal, float)
        self.t = 0
        self.path_len = 0.0
        self.workers = []
        for p in self.house.aisle_points(self.rng, self.n_workers, WORKER_R + 0.2):
            if np.linalg.norm(p - start) < 2.0:
                p = p + np.array([0.0, 2.5])
            self.workers.append(Worker(p, self._worker_target(p), self.rng.uniform(*WORKER_SPEED)))
        return self.observe()

    def _worker_target(self, p: np.ndarray) -> np.ndarray:
        """Workers walk in straight lines, so pick targets reachable without hitting racks."""
        for _ in range(50):
            q = self.house.aisle_points(self.rng, 1, WORKER_R + 0.2)[0]
            if self._segment_free(p, q, WORKER_R + 0.1):
                return q
        return p.copy()

    def _segment_free(self, a: np.ndarray, b: np.ndarray, margin: float) -> bool:
        n = max(2, int(np.linalg.norm(b - a) / 0.2))
        return all(self.house.clearance(a + (b - a) * k / n) >= margin for k in range(n + 1))

    # ---------------------------------------------------------------- sensing
    def lidar(self) -> np.ndarray:
        x, y, th = self.state
        ang = th + self.beam_rel
        d = np.stack([np.cos(ang), np.sin(ang)], 1)
        o = np.array([x, y])
        dist = np.full(LIDAR_BEAMS, LIDAR_RANGE)
        for r in self.house.obstacles:
            dist = np.minimum(dist, _ray_rect(o, d, r))
        for w in self.workers:
            dist = np.minimum(dist, _ray_circle(o, d, w.pos, WORKER_R))
        return np.clip(dist, 0.0, LIDAR_RANGE)

    def sensed_workers(self) -> list[np.ndarray]:
        p = self.state[:2]
        return [w.pos.copy() for w in self.workers if np.linalg.norm(w.pos - p) <= LIDAR_RANGE]

    def observe(self) -> dict:
        return {"state": self.state.copy(), "vel": self.vel.copy(), "goal": self.goal.copy(),
                "lidar": self.lidar(), "workers": self.sensed_workers()}

    # ---------------------------------------------------------------- dynamics
    def collided(self) -> bool:
        p = self.state[:2]
        if self.house.clearance(p) < ROBOT_R:
            return True
        return any(np.linalg.norm(w.pos - p) < ROBOT_R + WORKER_R for w in self.workers)

    def step(self, cmd) -> tuple[dict, dict]:
        """cmd = (v, omega) in physical units; clipped to limits and rate limits."""
        v, w = float(cmd[0]), float(cmd[1])
        v = np.clip(v, 0.0, V_MAX)
        w = np.clip(w, -W_MAX, W_MAX)
        dv = np.clip(v - self.vel[0], -A_MAX * DT, A_MAX * DT)
        dw = np.clip(w - self.vel[1], -ALPHA_MAX * DT, ALPHA_MAX * DT)
        self.vel = self.vel + np.array([dv, dw])

        x, y, th = self.state
        prev = np.array([x, y])
        th_new = th + self.vel[1] * DT
        x += self.vel[0] * np.cos(th + self.vel[1] * DT / 2) * DT
        y += self.vel[0] * np.sin(th + self.vel[1] * DT / 2) * DT
        self.state = np.array([x, y, (th_new + np.pi) % (2 * np.pi) - np.pi])
        self.path_len += float(np.linalg.norm(self.state[:2] - prev))

        for wk in self.workers:
            if wk.pause > 0:
                wk.pause -= DT
                continue
            to = wk.target - wk.pos
            dist = np.linalg.norm(to)
            if dist < 0.2:
                wk.pause = self.rng.uniform(0.0, 3.0)
                wk.target = self._worker_target(wk.pos)
                continue
            wk.pos = wk.pos + to / dist * min(wk.speed * DT, dist)

        self.t += 1
        reached = bool(np.linalg.norm(self.state[:2] - self.goal) < GOAL_TOL)
        crash = self.collided()
        timeout = self.t >= self.max_steps
        info = {"reached": reached, "collision": crash, "timeout": timeout and not reached,
                "t": self.t, "path_len": self.path_len}
        return self.observe(), info
