"""Policy-Gradient-Assisted MAP-Elites emitter for pyribs.

Implements the PGA part of PGA-MAP-Elites (Nilsson & Cully, 2021).  This
emitter maintains:

* A **replay buffer** of (obs, action, reward, next_obs, done) transitions
  collected from archive policies during evaluation.
* A **TD3-style critic** Q(s, a) trained on that replay buffer.
* At each ``ask()`` it samples elites from the archive and performs a few
  steps of deterministic policy-gradient ascent (maximizing Q) to produce
  improved candidate solutions.

The emitter plugs into pyribs' standard ``ask / tell`` interface — it does
**not** use ``ask_dqd / tell_dqd``.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ribs.archives import ArchiveBase
from ribs.emitters._emitter_base import EmitterBase

from policy import MLPPolicy, load_flat_params, flat_params, count_params


# ---------------------------------------------------------------------------
# Replay buffer  (GPU-resident, fixed-capacity ring buffer)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity ring buffer that lives entirely on one device."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int,
                 device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs      = torch.zeros(capacity, obs_dim,     device=device)
        self.action   = torch.zeros(capacity, action_dim,  device=device)
        self.reward   = torch.zeros(capacity,              device=device)
        self.next_obs = torch.zeros(capacity, obs_dim,     device=device)
        self.done     = torch.zeros(capacity,              device=device)
        self._ptr  = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def add(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
            next_obs: torch.Tensor, done: torch.Tensor) -> None:
        """Insert a batch of transitions (all tensors shaped ``(B, *)``).

        Wraps around when capacity is reached.
        """
        n = obs.shape[0]
        if n == 0:
            return
        idxs = torch.arange(self._ptr, self._ptr + n, device=self.device) % self.capacity
        self.obs[idxs]      = obs
        self.action[idxs]   = action
        self.reward[idxs]   = reward
        self.next_obs[idxs] = next_obs
        self.done[idxs]     = done
        self._ptr  = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Uniformly sample a mini-batch of transitions."""
        idxs = torch.randint(0, self._size, (batch_size,), device=self.device)
        return (self.obs[idxs], self.action[idxs], self.reward[idxs],
                self.next_obs[idxs], self.done[idxs])


# ---------------------------------------------------------------------------
# Twin-Q critic  (TD3-style, two independent Q networks)
# ---------------------------------------------------------------------------

class TwinCritic(nn.Module):
    """Two independent Q(s, a) networks for clipped double-Q learning."""

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.q1 = self._build_q(obs_dim, action_dim, hidden)
        self.q2 = self._build_q(obs_dim, action_dim, hidden)

    @staticmethod
    def _build_q(obs_dim: int, action_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = obs_dim + action_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1))


# ---------------------------------------------------------------------------
# PGA emitter
# ---------------------------------------------------------------------------

class PGAEmitter(EmitterBase):
    """Policy-gradient emitter for PGA-MAP-Elites.

    At each ``ask()`` it:
    1. Samples ``batch_size`` elites from the archive.
    2. For each elite, runs ``pg_steps`` of gradient ascent on Q₁(s, π(s))
       to produce an improved parameter vector.

    At each ``tell()`` it trains the twin critic on the replay buffer.

    Parameters
    ----------
    archive      : pyribs archive.
    obs_dim      : observation dimensionality.
    action_dim   : action dimensionality.
    policy_hidden: hidden-layer sizes for the MLP policy.
    batch_size   : solutions emitted per ``ask()``.
    pg_steps     : PG ascent steps per elite.
    actor_lr     : policy learning rate.
    critic_lr    : critic learning rate.
    gamma        : discount factor.
    tau          : Polyak-averaging coefficient for target critic.
    replay_capacity : maximum replay-buffer size.
    critic_batch : mini-batch size for critic updates.
    critic_updates : number of critic gradient steps per ``tell()``.
    device       : torch device string.
    seed         : RNG seed.
    """

    def __init__(
        self,
        archive: ArchiveBase,
        *,
        obs_dim: int,
        action_dim: int,
        policy_hidden: tuple[int, ...] = (64, 64),
        batch_size: int = 64,
        pg_steps: int = 10,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        replay_capacity: int = 1_000_000,
        critic_batch: int = 256,
        critic_updates: int = 300,
        device: str = "cuda",
        seed: int | None = None,
    ):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._policy_hidden = policy_hidden
        self._device = torch.device(device)

        # Template policy — used to infer solution_dim and for archive-action queries.
        template = MLPPolicy(obs_dim, action_dim, policy_hidden).to(self._device)
        solution_dim = count_params(template)
        self._template = template

        super().__init__(
            archive,
            solution_dim=solution_dim,
            bounds=None, lower_bounds=None, upper_bounds=None,
        )

        self._batch_size     = batch_size
        self._pg_steps       = pg_steps
        self._actor_lr       = actor_lr
        self._gamma          = gamma
        self._tau            = tau
        self._critic_updates = critic_updates
        self._critic_batch   = critic_batch
        self._rng = np.random.default_rng(seed)

        # Critic + frozen target copy.
        self._critic        = TwinCritic(obs_dim, action_dim).to(self._device)
        self._critic_target = copy.deepcopy(self._critic)
        self._critic_opt    = torch.optim.Adam(self._critic.parameters(), lr=critic_lr)

        # Replay buffer (all GPU).
        self._replay = ReplayBuffer(replay_capacity, obs_dim, action_dim, self._device)

    # -- pyribs interface ---------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def ask(self) -> np.ndarray:
        """Return ``batch_size`` improved solutions (flat param vectors)."""
        if self.archive.empty or self._replay.size < self._critic_batch:
            return self._random_solutions()
        elites = self.archive.sample_elites(self._batch_size)["solution"]
        return self._pg_improve(elites)

    def tell(self, solution, objective, measures, add_info, **fields) -> None:
        """Receive results and train the critic if enough data is available."""
        if self._replay.size >= self._critic_batch:
            self._train_critic()

    # -- replay buffer (called from the training loop) ----------------------

    def add_transitions(self, obs: torch.Tensor, actions: torch.Tensor,
                        rewards: torch.Tensor, next_obs: torch.Tensor,
                        dones: torch.Tensor) -> None:
        """Push a batch of GPU-resident transitions into the replay buffer."""
        self._replay.add(obs, actions, rewards, next_obs, dones)

    # -- internals ----------------------------------------------------------

    def _random_solutions(self) -> np.ndarray:
        """Emit randomly initialised policy parameters (warm-up phase)."""
        sols = np.stack([
            flat_params(MLPPolicy(self._obs_dim, self._action_dim,
                                  self._policy_hidden)).numpy()
            for _ in range(self._batch_size)
        ])
        return sols

    def _pg_improve(self, elite_params_np: np.ndarray) -> np.ndarray:
        """Improve each elite with a few steps of ∇_θ Q(s, π_θ(s)) ascent.

        Each elite is loaded into an independent policy, optimised for
        ``pg_steps`` Adam iterations on a fresh observation mini-batch each
        step, then flattened back.
        """
        B = elite_params_np.shape[0]
        results = np.empty_like(elite_params_np)

        for i in range(B):
            policy = MLPPolicy(
                self._obs_dim, self._action_dim, self._policy_hidden
            ).to(self._device)
            load_flat_params(
                policy,
                torch.as_tensor(elite_params_np[i], dtype=torch.float32,
                                device=self._device),
            )
            opt = torch.optim.Adam(policy.parameters(), lr=self._actor_lr)

            for _ in range(self._pg_steps):
                obs_b = self._replay.sample(self._critic_batch)[0]
                q_val = self._critic.q1_forward(obs_b, policy(obs_b))
                loss = -q_val.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            results[i] = flat_params(policy).cpu().numpy()

        return results

    def _train_critic(self) -> None:
        """Run ``critic_updates`` TD3-style gradient steps."""
        for _ in range(self._critic_updates):
            obs, action, reward, next_obs, done = \
                self._replay.sample(self._critic_batch)

            with torch.no_grad():
                next_action = self._archive_action(next_obs)
                tq1, tq2 = self._critic_target(next_obs, next_action)
                target_q = (reward.unsqueeze(-1)
                            + self._gamma * (1 - done.unsqueeze(-1))
                            * torch.min(tq1, tq2))

            q1, q2 = self._critic(obs, action)
            loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            self._critic_opt.zero_grad()
            loss.backward()
            self._critic_opt.step()

            # Polyak-average the target network.
            with torch.no_grad():
                for p, tp in zip(self._critic.parameters(),
                                 self._critic_target.parameters()):
                    tp.data.mul_(1 - self._tau).add_(p.data, alpha=self._tau)

    @torch.no_grad()
    def _archive_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute actions on ``obs`` using a random archive elite."""
        if self.archive.empty:
            return torch.zeros(obs.shape[0], self._action_dim, device=self._device)
        elite = self.archive.sample_elites(1)["solution"][0]
        load_flat_params(
            self._template,
            torch.as_tensor(elite, dtype=torch.float32, device=self._device),
        )
        return self._template(obs)
