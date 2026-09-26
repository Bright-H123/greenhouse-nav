"""Local controllers that follow a global path.

PurePursuit is the simple geometric baseline. The MPC and TD3 controllers live
in their own modules and share this interface:

    reset(sim)                      called at episode start
    act(obs, path) -> (v, omega)    one command per 0.1 s step
"""

from __future__ import annotations

import numpy as np

from nav.sim import V_MAX, W_MAX


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
