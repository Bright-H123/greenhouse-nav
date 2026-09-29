"""Render the same episode under several methods, as one GIF per method.

The seed is chosen by script, not by eye: --pick median takes the seed whose
completion time is closest to the median over a candidate block, so the GIF shows
a representative run rather than the best one. The caption says which rule was
used and what the seed was.

Run:
    python scripts/make_demo.py --methods rpp mpc td3_goal_s0 --workers 4 --pick median
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nav.runner import run_episode  # noqa: E402
from nav.sim import GreenhouseSim  # noqa: E402
from nav.world import Greenhouse  # noqa: E402
from scripts.evaluate import make  # noqa: E402


def pick_seed(methods: list[str], n_workers: int, aisle: float, seeds: list[int],
              rule: str) -> int:
    """A seed every method solves, chosen by the stated rule rather than by eye."""
    times: dict[int, list[float]] = {}
    for s in seeds:
        per = []
        for m in methods:
            ctrl, use_planner = make(m)
            sim = GreenhouseSim(Greenhouse(row_aisle=aisle), n_workers=n_workers, max_steps=900)
            info, _ = run_episode(sim, ctrl, seed=s, use_planner=use_planner)
            if not info["reached"]:
                per = []
                break
            per.append(info["t"] * 0.1)
        if per:
            times[s] = per
    if not times:
        raise SystemExit("no seed was solved by every method; widen --candidates")
    score = {s: float(np.mean(v)) for s, v in times.items()}
    order = sorted(score, key=score.get)
    chosen = order[len(order) // 2] if rule == "median" else order[0]
    print(f"{len(times)}/{len(seeds)} seeds solved by all methods; "
          f"{rule} seed {chosen}, mean time {score[chosen]:.1f} s")
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["rpp", "mpc"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aisle", type=float, default=3.0)
    ap.add_argument("--candidates", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=20_000)
    ap.add_argument("--pick", choices=["median", "fastest"], default="median")
    ap.add_argument("--seed", type=int, default=None, help="skip selection, use this seed")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    a = ap.parse_args()

    seeds = list(range(a.seed0, a.seed0 + a.candidates))
    seed = a.seed if a.seed is not None else pick_seed(a.methods, a.workers, a.aisle, seeds, a.pick)

    import imageio.v3 as iio
    a.out.mkdir(exist_ok=True)
    for m in a.methods:
        ctrl, use_planner = make(m)
        sim = GreenhouseSim(Greenhouse(row_aisle=a.aisle), n_workers=a.workers, max_steps=900)
        title = f"{getattr(ctrl, 'name', m)}   seed {seed}   {a.workers} workers"
        info, frames = run_episode(sim, ctrl, seed=seed, render=True, title=title,
                                   use_planner=use_planner, frame_every=2)
        out = a.out / f"demo_{m}_w{a.workers}.gif"
        iio.imwrite(out, frames, duration=1000 / a.fps, loop=0)
        outcome = ("reached" if info["reached"] else
                   "collision" if info["collision"] else "timeout")
        print(f"{m:14s} {outcome:9s} {info['t'] * 0.1:5.1f} s  {info['path_len']:5.1f} m  "
              f"{len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
