"""One episode loop shared by demos and evaluation.

The global planner replans every REPLAN_EVERY steps; the local controller acts
every step. Returns a result dict and, if requested, rendered frames.
"""

from __future__ import annotations

import time

import numpy as np

from nav.planner import GridPlanner
from nav.sim import GreenhouseSim

REPLAN_EVERY = 5   # 0.5 s at 10 Hz


def run_episode(sim: GreenhouseSim, controller, seed: int, render: bool = False,
                title: str = "", use_planner: bool = True, frame_every: int = 1):
    obs = sim.reset(seed=seed)
    controller.reset(sim)
    planner = GridPlanner(sim.house) if use_planner else None
    path: list = []
    if planner is not None:
        planner.reset(obs["state"][:2], sim.goal)
    frames, replans, t_plan, t_ctrl = [], 0, 0.0, 0.0
    min_clear = np.inf
    info = {}
    while True:
        if planner is not None and sim.t % REPLAN_EVERY == 0:
            t0 = time.perf_counter()
            path = planner.replan(obs["state"][:2], obs["workers"])
            t_plan += time.perf_counter() - t0
            replans += 1
        t0 = time.perf_counter()
        cmd = controller.act(obs, path)
        t_ctrl += time.perf_counter() - t0
        obs, info = sim.step(cmd)
        min_clear = min(min_clear, float(np.min(obs["lidar"])))
        if render and sim.t % frame_every == 0:
            from nav.render import draw
            hz = getattr(controller, "last_horizon", None)
            frames.append(draw(sim, path=path, horizon=hz,
                               title=f"{title}   t={sim.t * 0.1:5.1f} s",
                               dyn_mask=planner.dynamic if planner else None, planner=planner))
        if info["reached"] or info["collision"] or info["timeout"]:
            break
    info.update(replans=replans, plan_ms=1000 * t_plan / max(replans, 1),
                fallbacks=planner.fallbacks if planner else 0,
                mean_speed=info["path_len"] / max(sim.t * 0.1, 1e-9),
                ctrl_ms=1000 * t_ctrl / max(sim.t, 1), min_lidar=min_clear, seed=seed)
    return info, frames
