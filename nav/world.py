"""Greenhouse layout.

The default layout reproduces the Gazebo world `melon.world` from the original
2024-25 project: a 24 m x 19 m house with four rows of planting racks, each row a
left and a right rack of 7 m x 1 m, separated by a 4 m central aisle, with 3 m
aisles between rows. Rack and wall coordinates below were read from that file.

Two aisle widths are exposed as parameters so the same house can be made denser
for the aisle-width sweep. Rack size, row count and the outer walls stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# outer walls of melon.world (x in [-12, 12], y in [-9, 10])
X_MIN, X_MAX = -12.0, 12.0
Y_MIN, Y_MAX = -9.0, 10.0
RACK_LEN = 7.0          # along x
RACK_DEPTH = 1.0        # along y
ROWS = 4
ROW_CENTRE_Y = 0.5      # mean of the original rack centres 6.5, 2.5, -1.5, -5.5


@dataclass
class Rect:
    """Axis-aligned rectangle, [x0, x1] x [y0, y1]."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2])


@dataclass
class Greenhouse:
    row_aisle: float = 3.0      # gap between consecutive rows of racks
    centre_aisle: float = 4.0   # gap between the left and right rack of a row
    racks: list[Rect] = field(default_factory=list)
    walls: list[Rect] = field(default_factory=list)

    def __post_init__(self):
        pitch = RACK_DEPTH + self.row_aisle
        ys = ROW_CENTRE_Y + pitch * (np.arange(ROWS) - (ROWS - 1) / 2)
        half_gap = self.centre_aisle / 2
        self.racks = []
        for y in ys[::-1]:                       # top row first, like melon.world
            for side in (-1, 1):
                xin = side * half_gap
                xout = side * (half_gap + RACK_LEN)
                self.racks.append(Rect(min(xin, xout), y - RACK_DEPTH / 2,
                                       max(xin, xout), y + RACK_DEPTH / 2))
        t = 0.1
        self.walls = [
            Rect(X_MIN - t, Y_MAX, X_MAX + t, Y_MAX + t),   # north
            Rect(X_MIN - t, Y_MIN - t, X_MAX + t, Y_MIN),   # south
            Rect(X_MAX, Y_MIN, X_MAX + t, Y_MAX),           # east
            Rect(X_MIN - t, Y_MIN, X_MIN, Y_MAX),           # west
        ]

    @property
    def obstacles(self) -> list[Rect]:
        return self.racks + self.walls

    def occupancy(self) -> float:
        area = (X_MAX - X_MIN) * (Y_MAX - Y_MIN)
        return sum((r.x1 - r.x0) * (r.y1 - r.y0) for r in self.racks) / area

    def clearance(self, p: np.ndarray) -> float:
        """Distance from point p to the nearest static obstacle (0 if inside one)."""
        best = np.inf
        for r in self.obstacles:
            dx = max(r.x0 - p[0], 0.0, p[0] - r.x1)
            dy = max(r.y0 - p[1], 0.0, p[1] - r.y1)
            best = min(best, float(np.hypot(dx, dy)))
        return best

    def free(self, p: np.ndarray, margin: float) -> bool:
        return (X_MIN + margin <= p[0] <= X_MAX - margin
                and Y_MIN + margin <= p[1] <= Y_MAX - margin
                and self.clearance(p) >= margin)

    def aisle_points(self, rng: np.random.Generator, n: int, margin: float) -> np.ndarray:
        """Random free points, used for start/goal sampling and worker patrols."""
        out = []
        while len(out) < n:
            p = np.array([rng.uniform(X_MIN, X_MAX), rng.uniform(Y_MIN, Y_MAX)])
            if self.free(p, margin):
                out.append(p)
        return np.array(out)
