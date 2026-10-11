"""Fixed-seed evaluation of every method on every condition, in parallel.

Every method sees the same seeds, so start, goal, initial heading and worker
behaviour are identical across methods up to the point where the robot's own
motion changes what the workers do (they yield to it).

Usage:
    python scripts/evaluate.py --methods rpp mpc --workers 0 4 8 --seeds 100 --tag main
    python scripts/evaluate.py --methods rpp mpc --workers 4 --aisles 3.0 2.0 1.6 1.3 --tag aisles
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nav.stats import mcnemar_exact, wilson  # noqa: E402

_CACHE: dict = {}


def make(method: str):
    """One controller per process; the MPC solver takes a second to build."""
    if method not in _CACHE:
        if method == "pp":
            from nav.controllers import PurePursuit
            _CACHE[method] = (PurePursuit(), True)
        elif method == "rpp":
            from nav.controllers import RegulatedPurePursuit
            _CACHE[method] = (RegulatedPurePursuit(), True)
        elif method == "mpc":
            from nav.mpc import NMPC
            _CACHE[method] = (NMPC(), True)
        elif method == "mpc_nopred":
            from nav.mpc import NMPC
            _CACHE[method] = (NMPC(predict_workers=False), True)
        elif method.startswith("td3_"):
            # td3_<mode>_s<seed>, e.g. td3_goal_s0 or td3_waypoint_s1; loads
            # runs/<method>/best.pt. The end-to-end variant runs with no planner.
            import torch
            from nav.td3 import TD3, TD3Controller
            torch.set_num_threads(1)
            mode = "waypoint" if "waypoint" in method else "goal"
            ckpt = ROOT / "runs" / method / "best.pt"
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
            from nav.rl_env import N_BINS
            agent = TD3(N_BINS + 5, 2)
            agent.load(ckpt)
            _CACHE[method] = (TD3Controller(agent, mode), mode == "waypoint")
        else:
            raise ValueError(method)
    return _CACHE[method]


def run_one(job):
    method, aisle, n_workers, seed = job
    from nav.runner import run_episode
    from nav.sim import GreenhouseSim
    from nav.world import Greenhouse, rows_to_fill
    ctrl, use_planner = make(method)
    # a narrower aisle adds rows rather than opening a bypass, so the sweep
    # really does raise the density; rows_to_fill(3.0) is the default 4 rows
    house = Greenhouse(row_aisle=aisle, rows=rows_to_fill(aisle))
    sim = GreenhouseSim(house, n_workers=n_workers, max_steps=900)
    info, _ = run_episode(sim, ctrl, seed=seed, use_planner=use_planner)
    return {"method": method, "aisle": aisle, "rows": house.rows, "workers": n_workers, "seed": seed,
            "reached": bool(info["reached"]), "hit": info["hit"], "timeout": bool(info["timeout"]),
            "time": info["t"] * 0.1, "path_len": info["path_len"], "mean_speed": info["mean_speed"],
            "min_lidar": info["min_lidar"], "ctrl_ms": info["ctrl_ms"], "plan_ms": info["plan_ms"],
            "fallbacks": info["fallbacks"]}


def summarise(rows, methods):
    conds = sorted({(r["aisle"], r["workers"]) for r in rows}, key=lambda c: (-c[0], c[1]))
    head = (f"{'aisle':>5} {'wk':>3} {'method':<11} {'success':>8} {'95% CI':>13} {'rack':>5} "
            f"{'worker':>6} {'t/out':>5} {'time s':>7} {'speed':>6} {'clear':>6} {'ctrl ms':>7}")
    lines = [head, "-" * len(head)]
    for aisle, nw in conds:
        by = {m: sorted((r for r in rows if r["method"] == m and r["aisle"] == aisle
                         and r["workers"] == nw), key=lambda r: r["seed"]) for m in methods}
        for m in methods:
            rs = by[m]
            n, k = len(rs), sum(r["reached"] for r in rs)
            lo, hi = wilson(k, n)
            ok = [r for r in rs if r["reached"]]
            lines.append(
                f"{aisle:5.1f} {nw:3d} {m:<11} {k:3d}/{n:<4d} [{100*lo:4.0f}%,{100*hi:4.0f}%] "
                f"{sum(r['hit'] == 'static' for r in rs):5d} {sum(r['hit'] == 'worker' for r in rs):6d} "
                f"{sum(r['timeout'] for r in rs):5d} "
                f"{np.median([r['time'] for r in ok]) if ok else float('nan'):7.1f} "
                f"{np.median([r['mean_speed'] for r in ok]) if ok else float('nan'):6.2f} "
                f"{np.median([r['min_lidar'] for r in rs]):6.2f} "
                f"{np.mean([r['ctrl_ms'] for r in rs]):7.1f}")
        # paired success comparison of every method against the first one
        base = by[methods[0]]
        for m in methods[1:]:
            b = sum(x["reached"] and not y["reached"] for x, y in zip(base, by[m]))
            c = sum(y["reached"] and not x["reached"] for x, y in zip(base, by[m]))
            lines.append(f"{'':10}{m} vs {methods[0]}: only {methods[0]} {b}, only {m} {c}, "
                         f"exact McNemar p = {mcnemar_exact(b, c):.3g}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["rpp", "mpc"])
    ap.add_argument("--workers", nargs="+", type=int, default=[4])
    ap.add_argument("--aisles", nargs="+", type=float, default=[3.0])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=10_000)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()

    jobs = list(product(a.methods, a.aisles, a.workers, range(a.seed0, a.seed0 + a.seeds)))
    with Pool(a.procs) as pool:
        rows = pool.map(run_one, jobs, chunksize=4)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{a.tag}.json").write_text(json.dumps({"args": vars(a), "episodes": rows}, indent=1))
    table = summarise(rows, a.methods)
    (out / f"{a.tag}.txt").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
