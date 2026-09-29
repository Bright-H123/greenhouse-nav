"""TD3 (Fujimoto, van Hoof, Meger, ICML 2018), written for this repository.

TD3 is DDPG plus three corrections for the overestimation bias of a learned
critic. All three are here and each is marked in the code:

  1. Clipped double Q. Two critics are trained on the same target, and the target
     uses the smaller of the two. An overestimate in one critic no longer
     propagates, because the minimum is taken before bootstrapping.
  2. Delayed policy updates. The actor and the target networks are updated once
     every POLICY_DELAY critic updates, so the actor climbs a critic that has
     had time to settle.
  3. Target policy smoothing. Clipped Gaussian noise is added to the target
     action, which stops the actor from exploiting a sharp, spurious peak in the
     critic.

The networks are small on purpose: the observation is 41 numbers and training
runs on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = (256, 256)
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
POLICY_DELAY = 2
TARGET_NOISE = 0.2
NOISE_CLIP = 0.5


def mlp(sizes: tuple[int, ...], out: int, out_act: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers += [nn.Linear(sizes[-1], out)]
    if out_act is not None:
        layers += [out_act]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = mlp((obs_dim, *HIDDEN), act_dim, nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """Two independent Q networks in one module (trick 1)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.q1 = mlp((obs_dim + act_dim, *HIDDEN), 1)
        self.q2 = mlp((obs_dim + act_dim, *HIDDEN), 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], 1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.rew = np.zeros((size, 1), np.float32)
        self.done = np.zeros((size, 1), np.float32)
        self.size, self.ptr, self.full = size, 0, False

    def add(self, o, a, r, o2, done) -> None:
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, o2, float(done)
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self) -> int:
        return self.size if self.full else self.ptr

    def sample(self, batch: int, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
        idx = rng.integers(0, len(self), batch)
        return tuple(torch.as_tensor(x[idx]) for x in
                     (self.obs, self.act, self.rew, self.next_obs, self.done))


@dataclass
class TD3Stats:
    critic_loss: float = 0.0
    actor_loss: float = 0.0
    q_mean: float = 0.0


class TD3:
    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0, buffer: int = 300_000):
        torch.manual_seed(seed)
        self.actor = Actor(obs_dim, act_dim)
        self.actor_t = Actor(obs_dim, act_dim)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic = Critic(obs_dim, act_dim)
        self.critic_t = Critic(obs_dim, act_dim)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), ACTOR_LR)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), CRITIC_LR)
        self.buf = ReplayBuffer(obs_dim, act_dim, buffer)
        self.rng = np.random.default_rng(seed)
        self.updates = 0
        self.act_dim = act_dim

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise: float = 0.0) -> np.ndarray:
        a = self.actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        if noise > 0:
            a = a + self.rng.normal(0, noise, self.act_dim)
        return np.clip(a, -1.0, 1.0)

    def update(self, batch: int = 256) -> TD3Stats:
        o, a, r, o2, d = self.buf.sample(batch, self.rng)

        with torch.no_grad():
            # trick 3: smooth the target action
            noise = (torch.randn_like(a) * TARGET_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            a2 = (self.actor_t(o2) + noise).clamp(-1.0, 1.0)
            # trick 1: bootstrap from the smaller of the two target critics
            q1_t, q2_t = self.critic_t(o2, a2)
            target = r + GAMMA * (1.0 - d) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(o, a)
        closs = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_c.zero_grad(set_to_none=True)
        closs.backward()
        self.opt_c.step()

        stats = TD3Stats(critic_loss=float(closs.detach()), q_mean=float(target.mean()))
        self.updates += 1
        # trick 2: delay the actor and both target networks
        if self.updates % POLICY_DELAY == 0:
            aloss = -self.critic.q1(torch.cat([o, self.actor(o)], 1)).mean()
            self.opt_a.zero_grad(set_to_none=True)
            aloss.backward()
            self.opt_a.step()
            with torch.no_grad():
                for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - TAU).add_(TAU * p)
            stats.actor_loss = float(aloss.detach())
        return stats

    def save(self, path) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.actor.load_state_dict(ck["actor"])
        self.actor_t.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_t.load_state_dict(ck["critic"])


class TD3Controller:
    """Adapter so a trained policy runs in nav.runner beside the other controllers.

    It builds the observation from the same make_obs as training, using the path
    that runner passes in, so there is no second copy of the observation layout.
    """

    def __init__(self, agent: "TD3", mode: str, name: str | None = None):
        from nav.rl_env import target_point
        self.agent, self.mode = agent, mode
        self.name = name or ("D* + TD3" if mode == "waypoint" else "TD3 end-to-end")
        self._target_point = target_point

    def reset(self, sim) -> None:
        self.sim = sim

    def act(self, obs: dict, path):
        from nav.rl_env import make_obs
        tgt = self._target_point(self.mode, self.sim, path)
        v, w = self.agent.act(make_obs(obs, tgt))
        from nav.sim import V_MAX, W_MAX
        return float((v + 1.0) * 0.5 * V_MAX), float(w * W_MAX)
