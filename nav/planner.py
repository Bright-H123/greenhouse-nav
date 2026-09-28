"""Global planner: an inflated occupancy grid of the greenhouse plus D* Lite.

Static layer: racks and walls, inflated by the robot radius and a safety margin.
Beyond the inflated edge, a cost that decays exponentially with distance (as in the
ROS 2 Nav2 inflation layer) keeps paths near the aisle centre. Without it the
shortest path runs along the inflated edge, 0.5 m from the rack for a whole row.
Dynamic layer: workers the LiDAR currently sees, inflated by robot + worker radius.
Each replan updates only the cells whose status changed and lets D* Lite repair.

If the worker layer seals the robot in (a worker stands beside it and the inflated
rack is on the other side), D* Lite has no path and the robot would wait for as
long as the worker does. In that case the planner falls back to a second D* Lite
that holds the static layer only, and spacing to the worker is left to the local
controller, which sees the worker in its LiDAR.
"""

from __future__ import annotations

import numpy as np

from nav.dstar_lite import DStarLite
from nav.sim import ROBOT_R, WORKER_R
from nav.world import X_MAX, X_MIN, Y_MAX, Y_MIN, Greenhouse

RES = 0.2                 # m per cell
STATIC_MARGIN = 0.15
DYNAMIC_MARGIN = 0.25
PROX_WEIGHT = 2.0         # extra cost factor at the inflated edge
PROX_DECAY = 3.0          # 1/m, Nav2's default cost_scaling_factor


class GridPlanner:
    def __init__(self, house: Greenhouse, res: float = RES):
        self.house, self.res = house, res
        self.W = int(round((X_MAX - X_MIN) / res))
        self.H = int(round((Y_MAX - Y_MIN) / res))
        ii, jj = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        self.cx = X_MIN + (jj + 0.5) * res
        self.cy = Y_MIN + (ii + 0.5) * res
        inflate = ROBOT_R + STATIC_MARGIN
        clear = np.full((self.H, self.W), np.inf)
        for r in house.obstacles:
            dx = np.maximum.reduce([r.x0 - self.cx, np.zeros_like(self.cx), self.cx - r.x1])
            dy = np.maximum.reduce([r.y0 - self.cy, np.zeros_like(self.cy), self.cy - r.y1])
            clear = np.minimum(clear, np.hypot(dx, dy))
        self.static = clear < inflate
        self.penalty = PROX_WEIGHT * np.exp(-PROX_DECAY * np.maximum(clear - inflate, 0.0))
        self.static[0, :] = self.static[-1, :] = True
        self.static[:, 0] = self.static[:, -1] = True
        self.dstar: DStarLite | None = None
        self.dynamic = np.zeros_like(self.static)

    def to_cell(self, p) -> tuple[int, int]:
        j = int(np.clip((p[0] - X_MIN) / self.res, 0, self.W - 1))
        i = int(np.clip((p[1] - Y_MIN) / self.res, 0, self.H - 1))
        return (i, j)

    def to_world(self, c) -> np.ndarray:
        return np.array([self.cx[c], self.cy[c]])

    def _nearest_free(self, c, blocked):
        if not blocked[c]:
            return c
        free = np.argwhere(~blocked)
        k = np.argmin((free[:, 0] - c[0]) ** 2 + (free[:, 1] - c[1]) ** 2)
        return tuple(int(v) for v in free[k])

    def reset(self, start, goal) -> None:
        self.dynamic[:] = False
        s = self._nearest_free(self.to_cell(start), self.static)
        g = self._nearest_free(self.to_cell(goal), self.static)
        self.dstar = DStarLite(self.static, s, g, self.penalty)
        self.dstar.compute()
        self.static_dstar = DStarLite(self.static, s, g, self.penalty)
        self.static_dstar.compute()
        self.fallbacks = 0

    def _worker_mask(self, workers, robot_cell) -> np.ndarray:
        m = np.zeros_like(self.static)
        rad = ROBOT_R + WORKER_R + DYNAMIC_MARGIN
        for w in workers:
            m |= (self.cx - w[0]) ** 2 + (self.cy - w[1]) ** 2 < rad * rad
        m[robot_cell] = False      # never block the cell the robot is standing on
        return m

    def replan(self, pos, workers) -> list[np.ndarray]:
        """Move the search start to the robot, update the worker layer, repair, return a world path."""
        d = self.dstar
        cell = self._nearest_free(self.to_cell(pos), self.static)
        d.move_start(cell)
        new_dyn = self._worker_mask(workers, cell)
        # a cell freed by a worker leaving must stay blocked if it is also inside the
        # inflated static layer, so only cells outside the static layer are released
        d.set_cells(map(tuple, np.argwhere(new_dyn & ~self.dynamic & ~self.static)), True)
        d.set_cells(map(tuple, np.argwhere(~new_dyn & self.dynamic & ~self.static)), False)
        self.dynamic = new_dyn
        d.compute()
        cells = d.path()
        if not cells:
            self.fallbacks += 1
            self.static_dstar.move_start(cell)
            self.static_dstar.compute()
            cells = self.static_dstar.path()
        return [self.to_world(c) for c in cells]
