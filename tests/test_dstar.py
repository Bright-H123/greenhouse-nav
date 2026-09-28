"""D* Lite correctness against an independent Dijkstra.

The incremental repair is where D* Lite implementations go wrong, and a wrong one
does not crash: it returns a valid-looking but longer path. So every check here
compares path cost with a Dijkstra run from scratch on the same grid and the same
cost function, after blocking and after unblocking cells.

Run:  python -m pytest tests/test_dstar.py -q   (or  python tests/test_dstar.py)
"""

from __future__ import annotations

import heapq
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nav.dstar_lite import INF, DStarLite  # noqa: E402
from nav.planner import GridPlanner  # noqa: E402
from nav.world import Greenhouse  # noqa: E402


def dijkstra(ds: DStarLite, start, goal) -> float:
    dist = {start: 0.0}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            return d
        if d > dist.get(u, INF):
            continue
        for v in ds.neighbours(u):
            c = ds.cost(u, v)
            if c == INF:
                continue
            nd = d + c
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return INF


def path_cost(ds: DStarLite, path) -> float:
    return sum(ds.cost(a, b) for a, b in zip(path, path[1:]))


def check_valid(ds: DStarLite, path):
    assert path, "empty path"
    assert path[0] == ds.start and path[-1] == ds.goal
    for a, b in zip(path, path[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1, f"non-adjacent step {a}->{b}"
        assert ds.cost(a, b) < INF, f"step through blocked or corner-cut {a}->{b}"


def _setup(seed=0):
    house = Greenhouse()
    gp = GridPlanner(house)
    rng = np.random.default_rng(seed)
    free = np.argwhere(~gp.static)
    while True:
        s, g = (tuple(int(v) for v in free[rng.integers(len(free))]) for _ in range(2))
        if math.dist(s, g) > 60:
            return gp, s, g, rng


def _penalties(gp, rng):
    """Unit costs, the planner's proximity cost, and an arbitrary random cost field."""
    return {"unit": None, "proximity": gp.penalty, "random": rng.uniform(0, 3, gp.static.shape)}


def test_static_optimal():
    for seed, kind in itertools.product(range(5), ("unit", "proximity", "random")):
        gp, s, g, rng = _setup(seed)
        ds = DStarLite(gp.static, s, g, _penalties(gp, rng)[kind])
        ds.compute()
        p = ds.path()
        check_valid(ds, p)
        ref = dijkstra(ds, s, g)
        assert abs(path_cost(ds, p) - ref) < 1e-9 * ref, (seed, kind, path_cost(ds, p), ref)


def test_incremental_block_and_unblock():
    for seed, kind in itertools.product(range(5), ("unit", "proximity", "random")):
        gp, s, g, rng = _setup(seed)
        ds = DStarLite(gp.static, s, g, _penalties(gp, rng)[kind])
        ds.compute()
        p = ds.path()
        # drop a 5x5 blob onto the current path, well ahead of the robot
        c = p[len(p) // 2]
        blob = [(c[0] + di, c[1] + dj) for di in range(-2, 3) for dj in range(-2, 3)
                if (c[0] + di, c[1] + dj) not in (s, g)]
        blob = [b for b in blob if 0 <= b[0] < ds.H and 0 <= b[1] < ds.W and not gp.static[b]]
        # move the robot a few cells along the path first, as it would in use
        ds.move_start(p[5])
        ds.set_cells(blob, True)
        ds.compute()
        p2 = ds.path()
        check_valid(ds, p2)
        ref = dijkstra(ds, ds.start, g)
        assert abs(path_cost(ds, p2) - ref) < 1e-9 * ref, ("block", seed, kind)

        ds.set_cells(blob, False)
        ds.compute()
        p3 = ds.path()
        check_valid(ds, p3)
        ref = dijkstra(ds, ds.start, g)
        assert abs(path_cost(ds, p3) - ref) < 1e-9 * ref, ("unblock", seed, kind)


def test_planner_never_frees_static_cells():
    house = Greenhouse()
    gp = GridPlanner(house)
    gp.reset(np.array([0.0, 0.5]), np.array([10.5, 0.5]))
    before = gp.dstar.blocked.copy()
    # a worker standing against the rack at y in [2, 3], so its inflated disc overlaps
    # the rack's inflated static layer; without that overlap this test cannot fail
    worker = np.array([3.0, 3.5])
    gp.replan(np.array([0.0, 0.5]), [worker])
    overlap = gp._worker_mask([worker], gp.to_cell(np.array([0.0, 0.5]))) & gp.static
    assert overlap.any(), "test setup does not exercise the overlap case"
    gp.replan(np.array([0.0, 0.5]), [])            # worker leaves
    assert np.array_equal(gp.dstar.blocked, before), "static cells were released"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("pass", name)
