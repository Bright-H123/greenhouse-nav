"""Classify every failed episode: what was hit, how fast, and who was moving.

Usage: python scripts/diagnose.py [n_seeds] [n_workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nav.controllers import RegulatedPurePursuit  # noqa: E402
from nav.mpc import NMPC  # noqa: E402
from nav.planner import GridPlanner  # noqa: E402
from nav.runner import REPLAN_EVERY  # noqa: E402
from nav.sim import DT, ROBOT_R, WORKER_R, GreenhouseSim  # noqa: E402
from nav.world import Greenhouse  # noqa: E402


def trace(sim, ctrl, seed):
    obs = sim.reset(seed=seed)
    ctrl.reset(sim)
    planner = GridPlanner(sim.house)
    planner.reset(obs["state"][:2], sim.goal)
    path, log = [], []
    while True:
        if sim.t % REPLAN_EVERY == 0:
            path = planner.replan(obs["state"][:2], obs["workers"])
        cmd = ctrl.act(obs, path)
        prev_workers = [w.pos.copy() for w in sim.workers]
        prev_state = sim.state.copy()
        obs, info = sim.step(cmd)
        log.append((prev_state, prev_workers, sim.state.copy(), [w.pos.copy() for w in sim.workers],
                    sim.vel.copy(), len(path)))
        if info["reached"] or info["collision"] or info["timeout"]:
            return info, log, path


def describe(sim, info, log):
    s0, w0, s1, w1, vel, plen = log[-1]
    t = len(log) * DT
    if info["timeout"]:
        # where did it stall?
        pos = np.array([l[2][:2] for l in log])
        moved_last_10s = np.linalg.norm(pos[-1] - pos[-min(100, len(pos))])
        return (f"TIMEOUT at t={t:.1f}s, moved {moved_last_10s:.2f} m in last 10 s, "
                f"v={vel[0]:.2f}, goal dist {np.linalg.norm(s1[:2] - sim.goal):.1f} m, "
                f"static_clear={sim.house.clearance(s1[:2]):.2f}, path pts {plen}")
    if sim.house.clearance(s1[:2]) < ROBOT_R:
        return (f"RACK/WALL at t={t:.1f}s  v={vel[0]:.2f} w={vel[1]:.2f}  "
                f"static_clear={sim.house.clearance(s1[:2]):.2f}")
    k = int(np.argmin([np.linalg.norm(p - s1[:2]) for p in w1]))
    rel0 = w0[k] - s0[:2]
    u = rel0 / max(np.linalg.norm(rel0), 1e-9)
    robot_close = float((s1[:2] - s0[:2]) @ u) / DT
    worker_close = float(-(w1[k] - w0[k]) @ u) / DT
    return (f"WORKER at t={t:.1f}s  v={vel[0]:.2f}  robot closing {robot_close:.2f} m/s, "
            f"worker closing {worker_close:.2f} m/s, gap before {np.linalg.norm(rel0) - ROBOT_R - WORKER_R:.2f} m")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    for name, ctrl in [("RPP", RegulatedPurePursuit()), ("NMPC", NMPC())]:
        print(f"--- {name}")
        sim = GreenhouseSim(Greenhouse(), n_workers=nw, max_steps=900)
        for seed in range(n):
            info, log, _ = trace(sim, ctrl, seed)
            if not info["reached"]:
                print(f"   seed {seed:2d}: {describe(sim, info, log)}")


if __name__ == "__main__":
    main()
