# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-environment behavior descriptor definitions for QDax training.

Each environment needs:
  * A **state descriptor function** ``(obs_torch, env) → Tensor(N, desc_dim)``
    called every simulation step to build per-step descriptors.
  * A **state_descriptor_length** (int).

The episode-level descriptor extractor (mean over valid timesteps) is shared.
"""

from __future__ import annotations

import torch
import jax.numpy as jnp
from qdax.core.neuroevolution.buffers.buffer import QDTransition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_torch(x) -> torch.Tensor:
    """Convert warp or torch tensor → torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x
    import warp as wp
    return wp.to_torch(x)


# ---------------------------------------------------------------------------
# State descriptor functions  (obs, env) → Tensor(N, desc_dim)
# ---------------------------------------------------------------------------

def locomotion_xy_descriptor(obs: torch.Tensor, env) -> torch.Tensor:
    """Torso XY position (env-local), normalised to [0, 1].

    Works for Ant, Humanoid, and other locomotion DirectRLEnv envs where
    ``env.robot.data.root_pos_w`` gives the torso world position.
    """
    root_pos = _to_torch(env.robot.data.root_pos_w)
    origins = _to_torch(env.scene.env_origins)
    xy = root_pos[:, :2] - origins[:, :2]
    return ((xy + 10.0) / 20.0).clamp(0, 1)  # [-10, 10] → [0, 1]


def franka_ee_descriptor(obs: torch.Tensor, env) -> torch.Tensor:
    """Franka end-effector XYZ (env-local), normalised to [0, 1].

    The EE body is ``panda_hand`` on the ``robot`` articulation.
    """
    robot = env.scene["robot"]
    hand_idx = robot.body_names.index("panda_hand")
    body_pos = _to_torch(robot.data.body_pos_w)
    origins = _to_torch(env.scene.env_origins)
    ee_local = body_pos[:, hand_idx, :] - origins
    return ((ee_local + 1.0) / 2.0).clamp(0, 1)  # [-1, 1] → [0, 1]


def allegro_cube_descriptor(obs: torch.Tensor, env) -> torch.Tensor:
    """Allegro cube XYZ from the observation vector, normalised to [0, 1].

    In the ``"full"`` obs layout, indices 32:35 are the env-local cube position.
    """
    cube_xyz = obs[:, 32:35]
    return ((cube_xyz + 0.5) / 1.0).clamp(0, 1)  # [-0.5, 0.5] → [0, 1]


def cartpole_descriptor(obs: torch.Tensor, env) -> torch.Tensor:
    """Cart position + pole angle, normalised to [0, 1]."""
    cart_pos = obs[:, 0:1]
    pole_angle = torch.atan2(obs[:, 2:3], obs[:, 3:4])
    cart_desc = (cart_pos + 2.4) / 4.8
    pole_desc = (pole_angle + 0.21) / 0.42
    return torch.cat([cart_desc, pole_desc], dim=-1).clamp(0, 1)


# ---------------------------------------------------------------------------
# Episode-level descriptor extractor (shared by all envs)
# ---------------------------------------------------------------------------

def mean_descriptor_extractor(transitions: QDTransition, mask: jnp.ndarray) -> jnp.ndarray:
    """Mean state descriptor over non-masked timesteps."""
    valid = 1.0 - jnp.expand_dims(mask, axis=-1)
    return jnp.sum(transitions.state_desc * valid, axis=1) / jnp.sum(valid, axis=1).clip(min=1)


# ---------------------------------------------------------------------------
# Registry: task name → config dict
# ---------------------------------------------------------------------------

ENV_DESCRIPTORS = {
    "Isaac-Cartpole-Direct-v0": {
        "state_descriptor_fn": cartpole_descriptor,
        "state_descriptor_length": 2,
    },
    "Isaac-Ant-Direct-v0": {
        "state_descriptor_fn": locomotion_xy_descriptor,
        "state_descriptor_length": 2,
    },
    "Isaac-Humanoid-Direct-v0": {
        "state_descriptor_fn": locomotion_xy_descriptor,
        "state_descriptor_length": 2,
    },
    "Isaac-Reach-Franka-v0": {
        "state_descriptor_fn": franka_ee_descriptor,
        "state_descriptor_length": 3,
    },
    "Isaac-Repose-Cube-Allegro-Direct-v0": {
        "state_descriptor_fn": allegro_cube_descriptor,
        "state_descriptor_length": 3,
    },
}


def get_env_config(task: str) -> dict:
    """Return the descriptor config for *task*, raising if unknown."""
    if task not in ENV_DESCRIPTORS:
        raise KeyError(
            f"Unknown task {task!r}. Available: {list(ENV_DESCRIPTORS)}"
        )
    return ENV_DESCRIPTORS[task]
