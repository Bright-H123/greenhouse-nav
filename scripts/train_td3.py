"""Train a TD3 policy in the greenhouse, on CPU.

Two variants, selected with --mode:
    waypoint   local controller under D* Lite, target is the lookahead point
    goal       end-to-end, no global plan, target is the final goal

Training seeds and evaluation seeds are disjoint, and the evaluation seeds are
also disjoint from the seeds used to report final results, so nothing is selected
on the numbers that end up in the README.

    training      seeds drawn at random from [0, 100000)
    checkpoint    seeds 200000..200019, fixed, used to pick the saved policy
    reporting     seeds 10000+, never seen here (scripts/evaluate.py)

Run:
    python scripts/train_td3.py --mode goal --steps 300000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nav.rl_env import GreenhouseRLEnv  # noqa: E402
from nav.td3 import TD3  # noqa: E402
from nav.world import Greenhouse  # noqa: E402

CHECKPOINT_SEEDS = list(range(200_000, 200_020))
WARMUP = 10_000          # steps of uniform random actions before any update
EXPL_NOISE = 0.2         # std of the Gaussian added to the actor during training
EXPL_FINAL = 0.05
EXPL_DECAY_TO = 200_000


def evaluate(agent: TD3, mode: str, n_workers: int, seeds: list[int], aisle: float) -> dict:
    env = GreenhouseRLEnv(mode=mode, n_workers=n_workers, house=Greenhouse(row_aisle=aisle))
    reached = collisions = timeouts = 0
    returns, times = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        total = 0.0
        while True:
            obs, r, term, trunc, info = env.step(agent.act(obs))
            total += r
            if term or trunc:
                break
        returns.append(total)
        reached += int(info["reached"])
        collisions += int(info["collision"])
        timeouts += int(info["timeout"])
        if info["reached"]:
            times.append(info["t"] * 0.1)
    return {"reached": reached, "n": len(seeds), "collisions": collisions, "timeouts": timeouts,
            "return": float(np.mean(returns)),
            "time": float(np.mean(times)) if times else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["waypoint", "goal"], default="goal")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aisle", type=float, default=3.0)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    out = a.out or ROOT / "runs" / f"td3_{a.mode}_w{a.workers}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)

    env = GreenhouseRLEnv(mode=a.mode, n_workers=a.workers, house=Greenhouse(row_aisle=a.aisle))
    obs_dim = env.observation_space.shape[0]
    agent = TD3(obs_dim, 2, seed=a.seed)
    rng = np.random.default_rng(a.seed + 999)

    log_path = out / "train_log.jsonl"
    log = log_path.open("w", encoding="utf-8")
    best = -np.inf
    obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
    ep_ret, ep_len, episodes = 0.0, 0, 0
    recent: list[dict] = []
    t0 = time.perf_counter()

    for step in range(1, a.steps + 1):
        if step <= WARMUP:
            action = env.action_space.sample()
        else:
            frac = min(1.0, (step - WARMUP) / max(EXPL_DECAY_TO - WARMUP, 1))
            noise = EXPL_NOISE + frac * (EXPL_FINAL - EXPL_NOISE)
            action = agent.act(obs, noise=noise)

        obs2, reward, terminated, truncated, info = env.step(action)
        ep_ret += reward
        ep_len += 1
        # a timeout is not a terminal state for bootstrapping; only reaching the
        # goal or crashing is, so the value function is not told the episode
        # genuinely ended when it was merely cut short
        agent.buf.add(obs, action, reward, obs2, terminated)
        obs = obs2

        if terminated or truncated:
            episodes += 1
            recent.append({"ret": ep_ret, "len": ep_len, "reached": bool(info["reached"]),
                           "collision": bool(info["collision"])})
            obs, _ = env.reset(seed=int(rng.integers(0, 100_000)))
            ep_ret, ep_len = 0.0, 0

        if step > WARMUP:
            agent.update()

        if step % a.eval_every == 0:
            ev = evaluate(agent, a.mode, a.workers, CHECKPOINT_SEEDS, a.aisle)
            tail = recent[-50:]
            row = {"step": step, "episodes": episodes,
                   "train_return": float(np.mean([r["ret"] for r in tail])) if tail else None,
                   "train_reached": float(np.mean([r["reached"] for r in tail])) if tail else None,
                   "eval": ev, "minutes": round((time.perf_counter() - t0) / 60, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"{step:>7}  eval {ev['reached']:>2}/{ev['n']}  coll {ev['collisions']}  "
                  f"ret {ev['return']:7.1f}  train_ret "
                  f"{row['train_return'] if row['train_return'] is None else round(row['train_return'], 1)}"
                  f"  {row['minutes']:.1f} min", flush=True)
            # rank on successes first, then collisions, then return
            score = 1e6 * ev["reached"] - 1e3 * ev["collisions"] + ev["return"]
            if score > best:
                best = score
                agent.save(out / "best.pt")
                (out / "best.json").write_text(json.dumps(row, indent=1), encoding="utf-8")

    agent.save(out / "final.pt")
    log.close()
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
