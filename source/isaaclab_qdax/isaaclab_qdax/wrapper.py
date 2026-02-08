# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper that adapts an IsaacLab ``DirectRLEnv`` for use with QDax algorithms.

Key design decisions:

* **num_envs == batch_size** – each parallel IsaacLab env evaluates a
  different policy genotype, matching QDax's batch-evaluation model.
* **Zero-copy GPU transfers** – Torch ↔ JAX via DLPack (no CPU round-trip).
* **Pre-allocated buffers** – the rollout writes directly into fixed-size
  tensors instead of growing Python lists, reducing GC pressure.
* **JIT-compiled batched policy** – ``jax.vmap`` + ``jax.jit`` is applied
  once per rollout so the XLA trace is reused across timesteps.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.dlpack
import jax.numpy as jnp
import torch

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.custom_types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Params,
    RNGKey,
)

# ---------------------------------------------------------------------------
# Torch ↔ JAX zero-copy helpers (DLPack, GPU-only)
# ---------------------------------------------------------------------------


def torch_to_jax(t: torch.Tensor) -> jax.Array:
    """Zero-copy PyTorch CUDA → JAX GPU array via DLPack."""
    return jax.dlpack.from_dlpack(t.detach().contiguous())


def jax_to_torch(a: jax.Array, device: torch.device) -> torch.Tensor:
    """Zero-copy JAX GPU array → PyTorch CUDA tensor via DLPack."""
    return torch.from_dlpack(a).to(device)


# ---------------------------------------------------------------------------
# IsaacLab wrapper
# ---------------------------------------------------------------------------


class IsaacLabQDaxWrapper:
    """Adapter that lets QDax algorithms evaluate policies on an IsaacLab env.

    Does **not** subclass ``QDEnv`` because IsaacLab envs are imperative
    (they cannot be traced by ``jax.jit``).  Instead it exposes the
    properties that QDax emitters query and provides a :meth:`rollout`
    method that returns ``(Fitness, QDTransition)``.

    Args:
        env: An unwrapped IsaacLab ``DirectRLEnv``.
        state_descriptor_fn:
            ``(obs: Tensor[N, obs_dim], env) → Tensor[N, desc_dim]``.
            Called every step to build per-step state descriptors.
        state_descriptor_length: Dimensionality of the state descriptor.
    """

    def __init__(
        self,
        env,
        state_descriptor_fn: Callable[[torch.Tensor, object], torch.Tensor],
        state_descriptor_length: int,
    ) -> None:
        self.env = env
        self._state_desc_fn = state_descriptor_fn
        self._state_descriptor_length = state_descriptor_length
        self._device = torch.device(env.device)

    # -- Properties expected by QDax emitters ------------------------------

    @property
    def observation_size(self) -> int:
        return self.env.single_observation_space["policy"].shape[0]

    @property
    def action_size(self) -> int:
        return self.env.single_action_space.shape[0]

    @property
    def state_descriptor_length(self) -> int:
        return self._state_descriptor_length

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    # -- Core rollout ------------------------------------------------------

    def rollout(
        self,
        policy_params: Params,
        policy_fn: Callable[[Params, jax.Array], jax.Array],
        episode_length: int,
        key: RNGKey,
    ) -> Tuple[Fitness, QDTransition]:
        """Evaluate *num_envs* policies in parallel for *episode_length* steps.

        Returns:
            fitness: shape ``(num_envs,)`` – sum of rewards up to first done.
            transitions: ``QDTransition`` with leaves of shape
                ``(num_envs, episode_length, …)``.
        """
        n = self.num_envs
        dev = self._device
        obs_dim = self.observation_size
        act_dim = self.action_size
        sd_dim = self._state_descriptor_length

        # JIT-compile the vmapped policy *once* for this rollout so the XLA
        # trace is re-used across timesteps.
        batched_policy = jax.jit(jax.vmap(policy_fn, in_axes=(0, 0)))

        # Pre-allocate GPU buffers – (num_envs, episode_length, dim)
        buf_obs = torch.empty((n, episode_length, obs_dim), device=dev)
        buf_nobs = torch.empty_like(buf_obs)
        buf_act = torch.empty((n, episode_length, act_dim), device=dev)
        buf_rew = torch.zeros((n, episode_length), device=dev)
        buf_done = torch.zeros((n, episode_length), device=dev)
        buf_trunc = torch.zeros((n, episode_length), device=dev)
        buf_sd = torch.empty((n, episode_length, sd_dim), device=dev)
        buf_nsd = torch.empty_like(buf_sd)

        # Reset and get initial observation
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]
        sd = self._state_desc_fn(obs, self.env)

        # Track first-done per env (IsaacLab auto-resets internally).
        done_flag = torch.zeros(n, dtype=torch.bool, device=dev)

        for t in range(episode_length):
            # --- JAX: batched policy inference ---
            act_j = batched_policy(policy_params, torch_to_jax(obs))
            act = jax_to_torch(act_j, dev)

            # --- Torch: step the IsaacLab sim ---
            nobs_dict, rew, terminated, truncated, _ = self.env.step(act)
            nobs = nobs_dict["policy"]
            nsd = self._state_desc_fn(nobs, self.env)

            # Mask out rewards and done signals after first termination.
            rew_masked = rew.clone()
            rew_masked[done_flag] = 0.0
            step_done = (terminated | truncated) & ~done_flag

            buf_obs[:, t] = obs
            buf_nobs[:, t] = nobs
            buf_act[:, t] = act
            buf_rew[:, t] = rew_masked
            buf_done[:, t] = step_done.float()
            buf_trunc[:, t] = (truncated & ~done_flag).float()
            buf_sd[:, t] = sd
            buf_nsd[:, t] = nsd

            done_flag = done_flag | terminated | truncated
            obs, sd = nobs, nsd

        # Convert buffers to JAX (zero-copy) and build QDTransition.
        transitions = QDTransition(
            obs=torch_to_jax(buf_obs),
            next_obs=torch_to_jax(buf_nobs),
            rewards=torch_to_jax(buf_rew),
            dones=torch_to_jax(buf_done),
            truncations=torch_to_jax(buf_trunc),
            actions=torch_to_jax(buf_act),
            state_desc=torch_to_jax(buf_sd),
            next_state_desc=torch_to_jax(buf_nsd),
        )
        fitnesses = jnp.sum(transitions.rewards, axis=1)
        return fitnesses, transitions


# ---------------------------------------------------------------------------
# Scoring function builder
# ---------------------------------------------------------------------------


def make_isaaclab_scoring_fn(
    wrapper: IsaacLabQDaxWrapper,
    policy_fn: Callable[[Params, jax.Array], jax.Array],
    episode_length: int,
    descriptor_extractor: Callable[[QDTransition, jax.Array], Descriptor],
) -> Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]]:
    """Build a QDax-compatible scoring function backed by an IsaacLab env.

    Args:
        wrapper: The wrapped IsaacLab environment.
        policy_fn: Typically ``policy_network.apply``.
        episode_length: Max env steps per evaluation episode.
        descriptor_extractor: ``(transitions, mask) → descriptors``.

    Returns:
        ``(genotypes, key) → (fitness, descriptors, extra_scores)``
    """

    def scoring_fn(
        genotypes: Genotype, key: RNGKey,
    ) -> Tuple[Fitness, Descriptor, ExtraScores]:
        fitnesses, transitions = wrapper.rollout(
            policy_params=genotypes,
            policy_fn=policy_fn,
            episode_length=episode_length,
            key=key,
        )
        # Mask: 1 for timesteps *after* the first done, 0 otherwise.
        is_done = jnp.clip(jnp.cumsum(transitions.dones, axis=1), 0, 1)
        mask = jnp.roll(is_done, 1, axis=1).at[:, 0].set(0)

        descriptors = descriptor_extractor(transitions, mask)
        return fitnesses, descriptors, {"transitions": transitions}

    return scoring_fn
