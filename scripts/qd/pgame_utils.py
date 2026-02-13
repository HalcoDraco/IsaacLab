"""Utilities for integrating QDax PGA-ME with non-Brax (IsaacLab) environments.

PGA-ME's QualityPGEmitter expects a ``QDEnv`` argument, but only ever reads
three integer properties from it:

* ``observation_size``
* ``action_size``
* ``state_descriptor_length``

No ``env.step()``, ``env.reset()``, or any simulation method is called inside
the emitter — all training (TD3 critic/actor) happens purely on JAX replay
buffer data.  This means we do **not** need Brax at all; a lightweight shim
that provides these three properties is sufficient.

The emitter also requires ``extra_scores["transitions"]`` to contain a
``QDTransition`` object collected during the evaluation rollout.

This module provides:

1. ``IsaacLabQDEnvShim`` – duck-typed replacement for QDEnv (3 properties).
2. ``JaxTransitionEvaluator`` – extends ``JaxEvaluator`` to collect per-step
   ``QDTransition`` data during rollouts and return it via ``extra_scores``.
"""

from typing import Callable, Optional, Tuple

import torch
import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import QDTransition


# ---------------------------------------------------------------------------
# 1. Minimal duck-typed env shim (replaces QDEnv for the emitter)
# ---------------------------------------------------------------------------

class IsaacLabQDEnvShim:
    """Provides the 3 properties that ``QualityPGEmitter`` reads from ``QDEnv``.

    No Brax dependency.  Python duck-typing ensures this works even though the
    type annotation says ``QDEnv`` — at runtime, only attribute access happens.

    Args:
        observation_size: Dimensionality of the observation vector.
        action_size: Dimensionality of the action vector.
        state_descriptor_length: Dimensionality of per-step state descriptors.
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        state_descriptor_length: int,
    ):
        self._observation_size = observation_size
        self._action_size = action_size
        self._state_descriptor_length = state_descriptor_length

    @property
    def observation_size(self) -> int:
        return self._observation_size

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def state_descriptor_length(self) -> int:
        return self._state_descriptor_length


# ---------------------------------------------------------------------------
# 2. Transition-collecting evaluator (for PGA-ME's replay buffer)
# ---------------------------------------------------------------------------

class JaxTransitionEvaluator:
    """Evaluates batched Flax policies in IsaacLab **and** collects per-step
    ``QDTransition`` data required by PGA-ME.

    The returned ``extra_scores`` dict contains ``"transitions"`` — a
    ``QDTransition`` with shape ``(num_envs, episode_length, ...)`` for each
    field, exactly the format that ``QualityPGEmitter.state_update()`` expects
    (the replay buffer flattens batch × time internally).

    Args:
        env: IsaacLab gymnasium vectorized environment.
        num_envs: Number of parallel environments (= population size).
        policy_network: A Flax ``nn.Module`` (e.g. QDax ``MLP``).
        num_steps: Max steps per evaluation episode.
        descriptor_fn: ``(obs_accumulator, step_count, cumulative_rewards)
            -> descriptors (num_envs, D)``.  If *None*, defaults to time-
            averaged first 2 obs dims.
        state_descriptor_fn: ``(obs_t) -> state_desc (num_envs, sd_dim)``
            that extracts per-step state descriptors from the current
            observation tensor.  If *None*, defaults to ``obs[:, :2]``.
    """

    def __init__(
        self,
        env,
        num_envs: int,
        policy_network,
        num_steps: int = 100,
        descriptor_fn: Optional[Callable] = None,
        state_descriptor_fn: Optional[Callable] = None,
    ):
        self.env = env
        self.num_envs = num_envs
        self.policy_network = policy_network
        self.num_steps = num_steps
        self.descriptor_fn = descriptor_fn
        self.state_descriptor_fn = state_descriptor_fn
        self._batched_apply = jax.jit(jax.vmap(policy_network.apply))

    # ---- helpers ----
    def _get_obs(self, obs_dict):
        return obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    def _extract_state_desc(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-step state descriptor extraction (torch domain)."""
        if self.state_descriptor_fn is not None:
            return self.state_descriptor_fn(obs)
        return obs[:, :2]  # default: first 2 obs dims

    # ---- main entry point ----
    def evaluate(
        self, genotypes
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Roll out policies, collect transitions, return fitness + descriptors
        + ``extra_scores`` containing ``"transitions"``.

        Args:
            genotypes: JAX PyTree (each leaf leading dim = num_envs).

        Returns:
            fitnesses: ``(num_envs,)`` torch tensor.
            descriptors: ``(num_envs, D)`` torch tensor.
            extra_scores: ``{"transitions": QDTransition}`` with JAX arrays
                shaped ``(num_envs, episode_length, ...)``.
        """
        obs_dict, _ = self.env.reset()
        obs = self._get_obs(obs_dict)
        device = obs.device

        cumulative_rewards = torch.zeros(self.num_envs, device=device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        step_count = torch.zeros(self.num_envs, device=device)
        obs_accumulator = torch.zeros_like(obs)

        # Pre-allocate transition storage (torch, on GPU)
        obs_dim = obs.shape[-1]
        action_dim = self.env.action_space.shape[-1]
        sd_dim = self._extract_state_desc(obs).shape[-1]

        all_obs = torch.zeros(self.num_envs, self.num_steps, obs_dim, device=device)
        all_next_obs = torch.zeros_like(all_obs)
        all_actions = torch.zeros(self.num_envs, self.num_steps, action_dim, device=device)
        all_rewards = torch.zeros(self.num_envs, self.num_steps, device=device)
        all_dones = torch.zeros(self.num_envs, self.num_steps, device=device)
        all_truncations = torch.zeros(self.num_envs, self.num_steps, device=device)
        all_state_desc = torch.zeros(self.num_envs, self.num_steps, sd_dim, device=device)
        all_next_state_desc = torch.zeros_like(all_state_desc)

        actual_steps = 0

        with torch.no_grad():
            for t in range(self.num_steps):
                state_desc = self._extract_state_desc(obs)

                # Forward pass (JAX)
                obs_jax = jnp.from_dlpack(obs.contiguous())
                actions_jax = self._batched_apply(genotypes, obs_jax)
                actions = torch.from_dlpack(actions_jax)

                # Step env (PyTorch / IsaacLab)
                obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
                next_obs = self._get_obs(obs_dict)
                next_state_desc = self._extract_state_desc(next_obs)

                alive = ~dones

                # Store transition data
                all_obs[:, t] = obs
                all_next_obs[:, t] = next_obs
                all_actions[:, t] = actions
                all_rewards[:, t] = rewards * alive.float()
                all_dones[:, t] = (terminated | truncated).float()
                all_truncations[:, t] = truncated.float()
                all_state_desc[:, t] = state_desc
                all_next_state_desc[:, t] = next_state_desc

                # Accumulate fitness / descriptor stats
                cumulative_rewards += rewards * alive.float()
                obs_accumulator += next_obs * alive.float().unsqueeze(-1)
                step_count += alive.float()

                dones |= (terminated | truncated)
                actual_steps = t + 1

                if dones.all():
                    break

                obs = next_obs

        # Trim to actual episode length
        all_obs = all_obs[:, :actual_steps]
        all_next_obs = all_next_obs[:, :actual_steps]
        all_actions = all_actions[:, :actual_steps]
        all_rewards = all_rewards[:, :actual_steps]
        all_dones = all_dones[:, :actual_steps]
        all_truncations = all_truncations[:, :actual_steps]
        all_state_desc = all_state_desc[:, :actual_steps]
        all_next_state_desc = all_next_state_desc[:, :actual_steps]

        # Convert to JAX (zero-copy via DLPack)
        transitions = QDTransition(
            obs=jnp.from_dlpack(all_obs.contiguous()),
            next_obs=jnp.from_dlpack(all_next_obs.contiguous()),
            rewards=jnp.from_dlpack(all_rewards.contiguous()),
            dones=jnp.from_dlpack(all_dones.contiguous()),
            truncations=jnp.from_dlpack(all_truncations.contiguous()),
            actions=jnp.from_dlpack(all_actions.contiguous()),
            state_desc=jnp.from_dlpack(all_state_desc.contiguous()),
            next_state_desc=jnp.from_dlpack(all_next_state_desc.contiguous()),
        )

        # Compute episode-level descriptors
        if self.descriptor_fn is not None:
            descriptors = self.descriptor_fn(
                obs_accumulator, step_count, cumulative_rewards
            )
        else:
            mean_obs = obs_accumulator / step_count.unsqueeze(-1).clamp(min=1)
            descriptors = mean_obs[:, :2]

        return cumulative_rewards, descriptors, {"transitions": transitions}
