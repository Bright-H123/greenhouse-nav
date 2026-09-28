"""D* Lite (Koenig & Likhachev, 2002) on an 8-connected occupancy grid.

Searches backwards from the goal, so when the robot moves or cells change only
the affected part of the search is repaired instead of replanning from scratch.
That incremental repair is what makes it suited to a greenhouse where workers
walk into the aisle ahead of the robot.

Diagonal moves may not cut the corner of a blocked cell. A move into cell b costs
its length times (1 + penalty[b]); penalties are non-negative, so the octile
heuristic stays admissible and consistent.
"""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

INF = math.inf
SQ2 = math.sqrt(2.0)
NBRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class DStarLite:
    def __init__(self, blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
                 penalty: np.ndarray | None = None):
        self.blocked = blocked.copy()
        self.scale = 1.0 + (np.zeros(blocked.shape) if penalty is None else penalty)
        assert (self.scale >= 1.0).all(), "penalties must be non-negative"
        self.H, self.W = blocked.shape
        self.g = np.full(blocked.shape, INF)
        self.rhs = np.full(blocked.shape, INF)
        self.start, self.goal = start, goal
        self.last = start
        self.km = 0.0
        self._heap: list = []
        self._key: dict = {}
        self._tie = itertools.count()
        self.expanded = 0
        self.rhs[goal] = 0.0
        self._push(goal)

    # ----------------------------------------------------------------- helpers
    def _h(self, a, b) -> float:
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dx + dy) + (SQ2 - 2) * min(dx, dy)

    def _calc_key(self, s):
        m = min(self.g[s], self.rhs[s])
        return (m + self._h(self.start, s) + self.km, m)

    def _push(self, s):
        k = self._calc_key(s)
        self._key[s] = k
        heapq.heappush(self._heap, (k[0], k[1], next(self._tie), s))

    def _top(self):
        while self._heap:
            k0, k1, _, s = self._heap[0]
            if self._key.get(s) == (k0, k1):
                return (k0, k1), s
            heapq.heappop(self._heap)
        return (INF, INF), None

    def _inside(self, s) -> bool:
        return 0 <= s[0] < self.H and 0 <= s[1] < self.W

    def cost(self, a, b) -> float:
        if self.blocked[a] or self.blocked[b]:
            return INF
        di, dj = b[0] - a[0], b[1] - a[1]
        if di and dj:
            if self.blocked[a[0] + di, a[1]] or self.blocked[a[0], a[1] + dj]:
                return INF
            return SQ2 * self.scale[b]
        return self.scale[b]

    def neighbours(self, s):
        for di, dj in NBRS:
            n = (s[0] + di, s[1] + dj)
            if self._inside(n):
                yield n

    def _update_vertex(self, u):
        if u != self.goal:
            best = INF
            for s in self.neighbours(u):
                c = self.cost(u, s)
                if c < INF:
                    best = min(best, c + self.g[s])
            self.rhs[u] = best
        self._key.pop(u, None)
        if self.g[u] != self.rhs[u]:
            self._push(u)

    # ----------------------------------------------------------------- public
    def compute(self, max_expansions: int = 2_000_000) -> None:
        n = 0
        while True:
            k_old, u = self._top()
            if u is None:
                break
            if not (k_old < self._calc_key(self.start) or self.rhs[self.start] != self.g[self.start]):
                break
            n += 1
            if n > max_expansions:
                break
            k_new = self._calc_key(u)
            if k_old < k_new:
                self._push(u)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self._key.pop(u, None)
                for s in self.neighbours(u):
                    self._update_vertex(s)
            else:
                self.g[u] = INF
                self._update_vertex(u)
                for s in self.neighbours(u):
                    self._update_vertex(s)
        self.expanded += n

    def move_start(self, s) -> None:
        self.km += self._h(self.last, s)
        self.last = s
        self.start = s

    def set_cells(self, cells, value: bool) -> int:
        """Set cells blocked (True) or free (False); repair affected vertices."""
        touched = set()
        for c in cells:
            if self.blocked[c] != value:
                self.blocked[c] = value
                touched.add(c)
                touched.update(self.neighbours(c))
        for u in touched:
            self._update_vertex(u)
        return len(touched)

    def path(self, max_len: int = 100_000) -> list[tuple[int, int]]:
        if self.g[self.start] == INF:
            return []
        p, s = [self.start], self.start
        for _ in range(max_len):
            if s == self.goal:
                return p
            best, nxt = INF, None
            for n in self.neighbours(s):
                c = self.cost(s, n)
                if c < INF and c + self.g[n] < best:
                    best, nxt = c + self.g[n], n
            if nxt is None or nxt in p[-3:]:
                return []
            p.append(nxt)
            s = nxt
        return []
