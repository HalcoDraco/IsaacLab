"""Policy-Gradient-Assisted MAP-Elites emitter backed by Stable-Baselines3.

Implements PGA-MAP-Elites (Nilsson & Cully, 2021) by delegating the
reinforcement-learning components (replay buffer, critic training, target
networks) to an SB3 off-policy algorithm (TD3 or SAC).  This emitter only
adds the *policy-gradient improvement* step on top.

At each ``ask()`` it samples elites from the archive and runs a few steps
of deterministic policy-gradient ascent (maximising Q) to produce improved
candidate solutions.  At each ``tell()`` it trains the SB3 model's critic.

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

from policy import MLPPolicy, count_params, flat_params, load_flat_params

# Off-policy algorithms with a Q-function critic.
# On-policy algorithms (e.g. PPO) are incompatible: PGA-MAP-Elites needs an
# off-policy replay buffer and a Q(s,a) critic for policy-gradient ascent.
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
        template = MLPPolicy(obs_dim, action_dim, policy_hidden).to(self._device)
        solution_dim = count_params(template)  # scalar

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
        # Silence SB3 logging (logger is None by default → would crash on train).
        self._rl.set_logger(sb3_configure_logger(format_strings=[]))

    # -- pyribs interface ---------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def ask(self) -> np.ndarray:
        """Return ``batch_size`` improved solutions — shape ``(batch_size, param_dim)``."""
        if self.archive.empty or self._buffer_size() < self._critic_batch:
            return self._random_solutions()  # (batch_size, param_dim)
        elites = self.archive.sample_elites(self._batch_size)["solution"]
        return self._pg_improve(elites)  # (batch_size, param_dim)

    def tell(self, solution, objective, measures, add_info, **fields) -> None:
        """Receive results and train the SB3 critic."""
        if self._buffer_size() >= self._critic_batch:
            self._rl.train(
                gradient_steps=self._critic_updates,
                batch_size=self._critic_batch,
            )

    # -- replay buffer (called via transition_cb during evaluation) ---------

    def add_transitions(
        self,
        obs: torch.Tensor,       # (B, obs_dim)
        actions: torch.Tensor,   # (B, action_dim)
        rewards: torch.Tensor,   # (B,)
        next_obs: torch.Tensor,  # (B, obs_dim)
        dones: torch.Tensor,     # (B,)
    ) -> None:
        """Batch-write GPU-resident transitions into SB3's replay buffer."""
        n = obs.shape[0]
        if n == 0:
            return

        # GPU → CPU (negligible cost: a few KB per step).
        obs_np = obs.cpu().numpy()       # (B, obs_dim)
        act_np = actions.cpu().numpy()   # (B, action_dim)
        rew_np = rewards.cpu().numpy()   # (B,)
        nobs_np = next_obs.cpu().numpy() # (B, obs_dim)
        done_np = dones.cpu().numpy()    # (B,)

        # Direct batch insertion into the SB3 ring buffer.
        buf = self._rl.replay_buffer
        idxs = np.arange(buf.pos, buf.pos + n) % buf.buffer_size
        buf.observations[idxs, 0] = obs_np
        buf.next_observations[idxs, 0] = nobs_np
        buf.actions[idxs, 0] = act_np
        buf.rewards[idxs, 0] = rew_np
        buf.dones[idxs, 0] = done_np
        if hasattr(buf, "timeouts"):
            buf.timeouts[idxs, 0] = 0.0

        if buf.pos + n >= buf.buffer_size:
            buf.full = True
        buf.pos = (buf.pos + n) % buf.buffer_size

    # -- internals ----------------------------------------------------------

    def _buffer_size(self) -> int:
        """Current number of transitions in the SB3 replay buffer."""
        buf = self._rl.replay_buffer
        return buf.buffer_size if buf.full else buf.pos

    def _random_solutions(self) -> np.ndarray:
        """Emit randomly initialised policy parameters (warm-up)."""
        return np.stack([                                        # (batch_size, param_dim)
            flat_params(
                MLPPolicy(self._obs_dim, self._action_dim, self._policy_hidden)
            ).numpy()
            for _ in range(self._batch_size)
        ])

    def _pg_improve(self, elite_params_np: np.ndarray) -> np.ndarray:
        """Improve each elite via ∇_θ Q₁(s, π_θ(s)) ascent.

        For each elite:
        1. Load flat params into a fresh policy.
        2. Run ``pg_steps`` Adam iterations maximising Q₁.
        3. Flatten back to numpy.
        """
        B = elite_params_np.shape[0]          # batch_size
        results = np.empty_like(elite_params_np)  # (B, param_dim)

        for i in range(B):
            policy = MLPPolicy(
                self._obs_dim, self._action_dim, self._policy_hidden,
            ).to(self._device)
            load_flat_params(
                policy,
                torch.as_tensor(
                    elite_params_np[i],  # (param_dim,)
                    dtype=torch.float32,
                    device=self._device,
                ),
            )
            opt = torch.optim.Adam(policy.parameters(), lr=self._actor_lr)

            for _ in range(self._pg_steps):
                # Sample obs from SB3's buffer (already on self._device).
                data = self._rl.replay_buffer.sample(self._critic_batch)
                obs_b = data.observations           # (critic_batch, obs_dim)
                actions_b = policy(obs_b)            # (critic_batch, action_dim)
                q_val = self._rl.critic.q1_forward(  # (critic_batch, 1)
                    obs_b, actions_b,
                )
                loss = -q_val.mean()                 # scalar
                opt.zero_grad()
                loss.backward()
                opt.step()

            results[i] = flat_params(policy).cpu().numpy()  # (param_dim,)

        return results
