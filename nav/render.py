"""Top-down renderer for the greenhouse, OpenCV only."""

from __future__ import annotations

import cv2
import numpy as np

from nav.sim import LIDAR_RANGE, ROBOT_R, WORKER_R
from nav.world import X_MAX, X_MIN, Y_MAX, Y_MIN

PX = 26  # pixels per metre
W = int((X_MAX - X_MIN) * PX)
H = int((Y_MAX - Y_MIN) * PX)
BAR = 26  # title band above the map, so the title never covers the top aisle

# All colours are BGR, because OpenCV draws in BGR; draw() flips to RGB at the end.

FLOOR = (228, 232, 226)
RACK = (70, 118, 78)
RACK_EDGE = (40, 80, 48)
WALL = (90, 90, 90)
PATH = (220, 130, 40)       # blue
ROBOT = (120, 50, 20)       # dark navy
WORKER = (40, 140, 245)     # orange
GOAL = (70, 175, 70)        # green
RAY = (205, 200, 185)
HORIZON = (180, 60, 200)    # magenta


def to_px(p) -> tuple[int, int]:
    return (int((p[0] - X_MIN) * PX), int((Y_MAX - p[1]) * PX))


def draw(sim, path=None, horizon=None, title: str = "", dyn_mask=None, planner=None,
         show_rays: bool = True) -> np.ndarray:
    img = np.full((H, W, 3), FLOOR, np.uint8)

    if dyn_mask is not None and planner is not None and dyn_mask.any():
        for i, j in np.argwhere(dyn_mask):
            x0, y0 = to_px((X_MIN + j * planner.res, Y_MIN + (i + 1) * planner.res))
            cv2.rectangle(img, (x0, y0), (x0 + int(planner.res * PX), y0 + int(planner.res * PX)),
                          (215, 205, 245), -1)

    for r in sim.house.racks:
        a, b = to_px((r.x0, r.y1)), to_px((r.x1, r.y0))
        cv2.rectangle(img, a, b, RACK, -1)
        cv2.rectangle(img, a, b, RACK_EDGE, 1)
    for r in sim.house.walls:
        cv2.rectangle(img, to_px((r.x0, r.y1)), to_px((r.x1, r.y0)), WALL, -1)

    x, y, th = sim.state
    if show_rays:
        dist = sim.lidar()
        for a, d in zip(th + sim.beam_rel, dist):
            if d < LIDAR_RANGE - 1e-6:
                cv2.line(img, to_px((x, y)), to_px((x + d * np.cos(a), y + d * np.sin(a))), RAY, 1)

    if path is not None and len(path) > 1:
        pts = np.array([to_px(p) for p in path], np.int32)
        cv2.polylines(img, [pts], False, PATH, 2, cv2.LINE_AA)
    if horizon is not None and len(horizon) > 1:
        pts = np.array([to_px(p) for p in horizon], np.int32)
        cv2.polylines(img, [pts], False, HORIZON, 2, cv2.LINE_AA)

    cv2.circle(img, to_px(sim.goal), int(0.35 * PX), GOAL, -1, cv2.LINE_AA)
    for w in sim.workers:
        cv2.circle(img, to_px(w.pos), int(WORKER_R * PX), WORKER, -1, cv2.LINE_AA)
        cv2.circle(img, to_px(w.pos), int(WORKER_R * PX), (20, 90, 160), 1, cv2.LINE_AA)
    c = to_px((x, y))
    cv2.circle(img, c, int(ROBOT_R * PX), ROBOT, -1, cv2.LINE_AA)
    cv2.line(img, c, to_px((x + 0.5 * np.cos(th), y + 0.5 * np.sin(th))), (255, 255, 255), 2, cv2.LINE_AA)

    band = np.full((BAR, W, 3), 250, np.uint8)
    if title:
        cv2.putText(band, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    img = np.concatenate([band, img], 0)
    return np.ascontiguousarray(img[:, :, ::-1])  # BGR -> RGB for imageio
