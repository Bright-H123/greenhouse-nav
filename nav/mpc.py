"""Nonlinear MPC local controller (CasADi + IPOPT).

Tracks a reference sampled along the D* Lite path over a 1.5 s horizon on the
unicycle model, with speed, turn-rate and acceleration limits.

Two kinds of obstacle constraint:

  * LiDAR hits. The scan is split into angular sectors around the robot and the
    nearest hit in each sector becomes a disc. Taking the K nearest hits overall
    does not work: they all fall on the one rack face beside the robot and leave
    the path ahead unconstrained.
  * Detected workers, predicted forward with a constant-velocity model estimated
    from consecutive detections. This is the part a reactive controller cannot do.

Both are SOFT, because a hard constraint makes the problem infeasible the moment an
obstacle is already inside the margin, and an infeasible MPC gives no command at
all. Racks get an exact (L1) penalty far heavier than the one on workers. With one
quadratic weight for both, 0.15 m inside the rack margin cost less than 0.2 m
inside the larger worker margin, and the MPC dodged workers by grazing racks, the
wrong way round since racks never move and workers yield.

Solutions are warm-started from the previous one, shifted by a step.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from nav.controllers import lookahead_point
from nav.sim import A_MAX, ALPHA_MAX, DT, LIDAR_RANGE, ROBOT_R, V_MAX, W_MAX, WORKER_R

N = 15                 # horizon steps, 1.5 s
N_SECT = 16            # LiDAR sectors, 22.5 deg each; nearest hit per sector
SECT_RANGE = 3.0       # hits further than this are ignored
K_W = 4                # worker slots
OBS_R = 0.05           # radius assigned to a LiDAR hit point
SAFE = 0.10            # extra clearance to LiDAR hits
SAFE_W = 0.20          # extra clearance to predicted workers
W_VMAX = 1.5           # cap on an estimated worker speed, m/s
V_REF = 0.8            # same cruise speed as the RPP baseline
Q_POS, Q_HEAD, Q_TERM = 4.0, 0.5, 20.0
R_V, R_W = 0.1, 0.05
RD_V, RD_W = 1.0, 0.4
W1_STATIC, W2_STATIC = 1e4, 1e5     # linear + quadratic slack penalty, racks and walls
W1_WORKER, W2_WORKER = 50.0, 2000.0  # the same for predicted workers


class NMPC:
    name = "D* + MPC"

    def __init__(self, predict_workers: bool = True):
        self.predict_workers = predict_workers
        self._build()
        self.last_horizon = None

    def _build(self):
        opti = ca.Opti()
        X = opti.variable(3, N + 1)
        U = opti.variable(2, N)
        S = opti.variable(N_SECT + K_W, N)          # slack per obstacle per step
        x0 = opti.parameter(3)
        u_prev = opti.parameter(2)
        ref = opti.parameter(3, N + 1)
        obs = opti.parameter(2, N_SECT)
        obs_on = opti.parameter(N_SECT)             # 1 if this sector has a hit
        wk = opti.parameter(2, K_W * N)             # predicted worker centre, slot j at step k
        wk_on = opti.parameter(K_W)

        cost = 0
        r_obs = ROBOT_R + OBS_R + SAFE
        r_wk = ROBOT_R + WORKER_R + SAFE_W
        for k in range(N):
            th = X[2, k]
            nxt = ca.vertcat(X[0, k] + U[0, k] * ca.cos(th) * DT,
                             X[1, k] + U[0, k] * ca.sin(th) * DT,
                             X[2, k] + U[1, k] * DT)
            opti.subject_to(X[:, k + 1] == nxt)
            e = X[:2, k + 1] - ref[:2, k + 1]
            cost += Q_POS * ca.sumsqr(e) + Q_HEAD * (1 - ca.cos(X[2, k + 1] - ref[2, k + 1]))
            cost += R_V * U[0, k] ** 2 + R_W * U[1, k] ** 2
            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
            cost += RD_V * du[0] ** 2 + RD_W * du[1] ** 2
            opti.subject_to(opti.bounded(-A_MAX * DT, du[0], A_MAX * DT))
            opti.subject_to(opti.bounded(-ALPHA_MAX * DT, du[1], ALPHA_MAX * DT))
            p = X[:2, k + 1]
            for j in range(N_SECT):
                opti.subject_to(obs_on[j] * (r_obs ** 2 - ca.sumsqr(p - obs[:, j])) <= S[j, k])
            for j in range(K_W):
                c = wk[:, j * N + k]
                opti.subject_to(wk_on[j] * (r_wk ** 2 - ca.sumsqr(p - c)) <= S[N_SECT + j, k])
            s_st, s_wk = S[:N_SECT, k], S[N_SECT:, k]
            cost += W1_STATIC * ca.sum1(s_st) + W2_STATIC * ca.sumsqr(s_st)
            cost += W1_WORKER * ca.sum1(s_wk) + W2_WORKER * ca.sumsqr(s_wk)
        cost += Q_TERM * ca.sumsqr(X[:2, N] - ref[:2, N])

        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(opti.bounded(0.0, U[0, :], V_MAX))
        opti.subject_to(opti.bounded(-W_MAX, U[1, :], W_MAX))
        opti.subject_to(ca.vec(S) >= 0)
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"print_level": 0, "max_iter": 80, "tol": 1e-4,
                     "acceptable_tol": 1e-3, "warm_start_init_point": "yes", "sb": "yes"})
        self.opti, self.X, self.U, self.S = opti, X, U, S
        self.p = dict(x0=x0, u_prev=u_prev, ref=ref, obs=obs, obs_on=obs_on, wk=wk, wk_on=wk_on)

    def reset(self, sim) -> None:
        self.u_prev = np.zeros(2)
        self.X_guess = None
        self.U_guess = np.zeros((2, N))
        self.prev_workers: list[np.ndarray] = []
        self.fails = 0
        self.solves = 0

    def _reference(self, state, path):
        """ref[:, k] is where the robot should be after k steps; ref[:, 0] is its projection."""
        pos = state[:2]
        pts = [lookahead_point(pos, path, V_REF * DT * k) for k in range(N + 1)]
        ref = np.zeros((3, N + 1))
        head_prev = state[2]
        for k in range(N + 1):
            a = pts[k] if k == 0 else pts[k - 1]
            b = pts[k + 1] if k < N else pts[k]
            d = b - a
            head = np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 1e-4 else head_prev
            head = head_prev + ((head - head_prev + np.pi) % (2 * np.pi) - np.pi)   # unwrap
            ref[:, k] = [pts[k][0], pts[k][1], head]
            head_prev = head
        return ref

    def _sector_hits(self, obs):
        x, y, th = obs["state"]
        dist, rel = obs["lidar"], obs["lidar_rel"]
        sect = ((rel + np.pi) / (2 * np.pi) * N_SECT).astype(int) % N_SECT
        P, on = np.zeros((2, N_SECT)), np.zeros(N_SECT)
        for j in range(N_SECT):
            m = (sect == j) & (dist < min(SECT_RANGE, LIDAR_RANGE - 1e-6))
            if not m.any():
                continue
            i = np.flatnonzero(m)[np.argmin(dist[m])]
            a = th + rel[i]
            P[:, j] = [x + dist[i] * np.cos(a), y + dist[i] * np.sin(a)]
            on[j] = 1.0
        return P, on

    def _worker_predictions(self, obs):
        pos = obs["state"][:2]
        now = sorted((np.asarray(w) for w in obs["workers"]), key=lambda w: np.linalg.norm(w - pos))
        W, on = np.zeros((2, K_W * N)), np.zeros(K_W)
        for j, w in enumerate(now[:K_W]):
            vel = np.zeros(2)
            if self.predict_workers and self.prev_workers:
                d = [np.linalg.norm(w - q) for q in self.prev_workers]
                i = int(np.argmin(d))
                if d[i] < W_VMAX * DT:                      # same worker, one step apart
                    vel = (w - self.prev_workers[i]) / DT
            for k in range(N):
                W[:, j * N + k] = w + vel * DT * (k + 1)
            on[j] = 1.0
        self.prev_workers = [np.asarray(w) for w in obs["workers"]]
        return W, on

    def act(self, obs: dict, path):
        state = obs["state"]
        if not path:
            self.prev_workers = [np.asarray(w) for w in obs["workers"]]
            self.u_prev = np.zeros(2)
            return 0.0, 0.0
        ref = self._reference(state, path)
        P, on = self._sector_hits(obs)
        Wp, won = self._worker_predictions(obs)
        o, p = self.opti, self.p
        o.set_value(p["x0"], state)
        o.set_value(p["u_prev"], self.u_prev)
        o.set_value(p["ref"], ref)
        o.set_value(p["obs"], P)
        o.set_value(p["obs_on"], on)
        o.set_value(p["wk"], Wp)
        o.set_value(p["wk_on"], won)
        o.set_initial(self.X, self.X_guess if self.X_guess is not None else ref)
        o.set_initial(self.U, self.U_guess)
        o.set_initial(self.S, 0)
        self.solves += 1
        try:
            sol = o.solve()
            Xs, Us = sol.value(self.X), sol.value(self.U)
        except RuntimeError:
            # fall back to the solver's last iterate rather than stopping dead
            self.fails += 1
            Xs, Us = o.debug.value(self.X), o.debug.value(self.U)
        self.X_guess = np.hstack([Xs[:, 1:], Xs[:, -1:]])
        self.U_guess = np.hstack([Us[:, 1:], Us[:, -1:]])
        self.last_horizon = [Xs[:2, k] for k in range(N + 1)]
        u = np.array([np.clip(Us[0, 0], 0, V_MAX), np.clip(Us[1, 0], -W_MAX, W_MAX)])
        self.u_prev = u
        return float(u[0]), float(u[1])
