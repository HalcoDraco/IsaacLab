"""Policy-Gradient-Assisted MAP-Elites emitter backed by Stable-Baselines3.

Implements PGA-MAP-Elites (Nilsson & Cully, 2021) by delegating the
reinforcement-learning components (replay buffer, critic training, target
networks) to an SB3 off-policy algorithm (TD3 or SAC).  This emitter only
adds the *policy-gradient improvement* step on top.

At each ``ask()`` it samples elites from the archive and runs a few steps
of deterministic policy-gradient ascent (maximising Q) to produce improved
candidate solutions.  At each ``tell()`` it trains the SB3 model's critic.

The PG improvement is **batched**: all elites are improved simultaneously
using a single shared critic, avoiding the sequential per-elite loop that
was the main bottleneck.

The emitter plugs into pyribs' standard ``ask / tell`` interface.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ribs.archives import ArchiveBase
from ribs.emitters._emitter_base import EmitterBase
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.logger import configure as sb3_configure_logger
from stable_baselines3.common.vec_env import DummyVecEnv

from policy import MLPPolicy, count_params, flat_params

# Off-policy algorithms with a Q-function critic.
_RL_ALGORITHMS: dict[str, type] = {"TD3": TD3, "SAC": SAC}


# ---------------------------------------------------------------------------
# Dummy env (SB3 requires an env for model initialisation)
# ---------------------------------------------------------------------------

def _make_dummy_env(obs_space: spaces.Box, act_space: spaces.Box) -> DummyVecEnv:
    """Create a no-op vectorised env so SB3 can set up its networks."""

    class _Env(gym.Env):
        observation_space = obs_space
        action_space = act_space

        def reset(self, **kw):
            return np.zeros(obs_space.shape, np.float32), {}

        def step(self, a):
            return np.zeros(obs_space.shape, np.float32), 0.0, True, False, {}

    return DummyVecEnv([_Env])


# ---------------------------------------------------------------------------
# PGA emitter
# ---------------------------------------------------------------------------

class PGAEmitter(EmitterBase):
    """Policy-gradient emitter backed by an SB3 off-policy algorithm.

    Parameters
    ----------
    archive       : pyribs archive.
    obs_dim       : observation dimensionality.
    action_dim    : action dimensionality.
    policy_hidden : hidden-layer sizes for the *archive* MLP policy.
    batch_size    : solutions emitted per ``ask()``.
    pg_steps      : PG ascent steps per elite.
    rl_algorithm  : ``"TD3"`` or ``"SAC"`` — which SB3 algorithm to use for
                    critic training.
    rl_kwargs     : extra keyword arguments forwarded to the SB3 model
                    constructor (learning_rate, gamma, tau, …).
    actor_lr      : learning rate for PG improvement of archive policies.
    replay_capacity : maximum transitions stored by SB3's replay buffer.
    critic_batch  : mini-batch size for SB3 critic training.
    critic_updates: gradient steps on the critic per ``tell()``.
    device        : torch device string.
    seed          : RNG seed.
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
        rl_algorithm: str = "TD3",
        rl_kwargs: dict | None = None,
        actor_lr: float = 3e-4,
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

        # Template policy — only used for param counting + PG improvement.
        self._template = MLPPolicy(obs_dim, action_dim, policy_hidden).to(self._device)
        solution_dim = count_params(self._template)

        # Cache parameter structure and build vmapped forward for PG improvement.
        self._param_names = [n for n, _ in self._template.named_parameters()]
        self._param_shapes = [p.shape for p in self._template.parameters()]
        self._param_numel = [p.numel() for p in self._template.parameters()]
        self._pg_batched_fwd = self._make_pg_batched_fwd()

        super().__init__(
            archive,
            solution_dim=solution_dim,
            bounds=None,
            lower_bounds=None,
            upper_bounds=None,
        )

        self._batch_size = batch_size
        self._pg_steps = pg_steps
        self._actor_lr = actor_lr
        self._critic_updates = critic_updates
        self._critic_batch = critic_batch
        self._rng = np.random.default_rng(seed)

        # ── SB3 off-policy model ──────────────────────────────────────────
        if rl_algorithm not in _RL_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{rl_algorithm}'. "
                f"Choose from {list(_RL_ALGORITHMS)}."
            )

        obs_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, (action_dim,), dtype=np.float32)

        sb3_kwargs: dict = dict(
            learning_rate=3e-4,
            buffer_size=replay_capacity,
            batch_size=critic_batch,
            gamma=0.99,
            tau=0.005,
            verbose=0,
            device=device,
            seed=seed,
        )
        if rl_kwargs:
            sb3_kwargs.update(rl_kwargs)

        rl_cls = _RL_ALGORITHMS[rl_algorithm]
        self._rl = rl_cls(
            "MlpPolicy",
            _make_dummy_env(obs_space, act_space),
            **sb3_kwargs,
        )
        # Silence SB3 logging.
        self._rl.set_logger(sb3_configure_logger(format_strings=[]))

    # -- pyribs interface ---------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def ask(self) -> np.ndarray:
        """Return ``batch_size`` improved solutions."""
        if self.archive.empty or self._buffer_size() < self._critic_batch:
            return self._random_solutions()
        elites = self.archive.sample_elites(self._batch_size)["solution"]
        return self._pg_improve(elites)

    def tell(self, solution, objective, measures, add_info, **fields) -> None:
        """Receive results and train the SB3 critic."""
        if self._buffer_size() >= self._critic_batch:
            self._rl.train(
                gradient_steps=self._critic_updates,
                batch_size=self._critic_batch,
            )

    # -- replay buffer (called via transition_cb after rollout) -------------

    def add_transitions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Bulk-write GPU-resident transitions into SB3's replay buffer.

        Called **once per generation** with all first-episode transitions
        collected during the GPU rollout.
        """
        n = obs.shape[0]
        if n == 0:
            return

        # Single GPU→CPU transfer of all transitions at once.
        obs_np = obs.cpu().numpy()
        act_np = actions.cpu().numpy()
        rew_np = rewards.cpu().numpy()
        nobs_np = next_obs.cpu().numpy()
        done_np = dones.cpu().numpy()

        # Direct batch insertion into the SB3 ring buffer.
        buf = self._rl.replay_buffer
        # Process in chunks that fit within the ring buffer.
        remaining = n
        src_offset = 0
        while remaining > 0:
            space = buf.buffer_size - buf.pos
            chunk = min(remaining, space)
            end = buf.pos + chunk
            sl = slice(src_offset, src_offset + chunk)
            buf.observations[buf.pos:end, 0] = obs_np[sl]
            buf.next_observations[buf.pos:end, 0] = nobs_np[sl]
            buf.actions[buf.pos:end, 0] = act_np[sl]
            buf.rewards[buf.pos:end, 0] = rew_np[sl]
            buf.dones[buf.pos:end, 0] = done_np[sl]
            if hasattr(buf, "timeouts"):
                buf.timeouts[buf.pos:end, 0] = 0.0
            buf.pos = (buf.pos + chunk) % buf.buffer_size
            if end >= buf.buffer_size:
                buf.full = True
            src_offset += chunk
            remaining -= chunk

    # -- internals ----------------------------------------------------------

    def _buffer_size(self) -> int:
        """Current number of transitions in the SB3 replay buffer."""
        buf = self._rl.replay_buffer
        return buf.buffer_size if buf.full else buf.pos

    def _random_solutions(self) -> np.ndarray:
        """Emit randomly initialised policy parameters (warm-up).

        Uses a single template and generates random weights in bulk via
        torch instead of creating individual MLPPolicy instances.
        """
        return np.stack([
            flat_params(
                MLPPolicy(self._obs_dim, self._action_dim, self._policy_hidden)
            ).numpy()
            for _ in range(self._batch_size)
        ])

    def _pg_improve(self, elite_params_np: np.ndarray) -> np.ndarray:
        """Improve elites via batched ∇_θ Q₁(s, π_θ(s)) ascent.

        Uses ``torch.vmap`` over the flat parameter tensor so all elites
        are forward-passed through the policy and critic in a single fused
        GPU operation per PG step, avoiding the sequential per-elite loop.
        """
        B = elite_params_np.shape[0]

        # All elite params as a single (B, param_dim) leaf tensor.
        all_params = torch.tensor(
            elite_params_np, dtype=torch.float32, device=self._device,
        ).requires_grad_(True)

        opt = torch.optim.Adam([all_params], lr=self._actor_lr)

        for _ in range(self._pg_steps):
            data = self._rl.replay_buffer.sample(self._critic_batch)
            obs_b = data.observations  # (S, obs_dim)
            S = obs_b.shape[0]

            # Batched policy forward: (B, S, act_dim).
            all_actions = self._pg_batched_fwd(all_params, obs_b)

            # Expand obs to match, then single big critic forward.
            obs_flat = obs_b.unsqueeze(0).expand(B, -1, -1).reshape(B * S, -1)
            act_flat = all_actions.reshape(B * S, -1)
            q_vals = self._rl.critic.q1_forward(obs_flat, act_flat)  # (B*S, 1)
            loss = -q_vals.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        return all_params.detach().cpu().numpy()

    def _make_pg_batched_fwd(self):
        """Build a vmapped forward for batched PG improvement.

        The returned callable maps ``(B, param_dim), (S, obs_dim) → (B, S, act_dim)``
        by vmapping over the first (policy batch) dimension only.
        """
        template = self._template
        names = self._param_names
        shapes = self._param_shapes
        numel = self._param_numel

        def _unflatten(flat: torch.Tensor) -> dict[str, torch.Tensor]:
            params: dict[str, torch.Tensor] = {}
            offset = 0
            for n, s, c in zip(names, shapes, numel):
                params[n] = flat[offset : offset + c].reshape(s)
                offset += c
            return params

        def _single_fwd(flat_p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
            return torch.func.functional_call(template, _unflatten(flat_p), obs)

        return torch.vmap(_single_fwd, in_dims=(0, None))
