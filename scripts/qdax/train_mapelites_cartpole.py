# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MAP-Elites on IsaacLab Cartpole via the ``isaaclab_qdax`` wrapper.

Usage (inside the IsaacLab Docker container)::

    source scripts/qdax/setup_jax_cuda.sh
    ./isaaclab.sh -p scripts/qdax/train_mapelites_cartpole.py \\
        --num_envs 100 --episode_length 200 --num_iterations 100
"""

from __future__ import annotations

import functools
import os
import sys
import time

# isaaclab_qdax is not pip-installed; add it to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/isaaclab_qdax"))

import argparse

import jax
import jax.numpy as jnp
import torch

# -- IsaacLab bootstrap (must precede any IsaacLab imports) ----------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="QDax MAP-Elites on IsaacLab Cartpole.",
)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--episode_length", type=int, default=200)
parser.add_argument("--num_iterations", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -- IsaacLab + QDax imports (after bootstrap) -----------------------------
import gymnasium as gym

import isaaclab_tasks  # noqa: F401 – registers gym envs
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_qdax import IsaacLabQDaxWrapper, make_isaaclab_scoring_fn

from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import default_qd_metrics


# -----------------------------------------------------------------------
# Descriptor helpers
# -----------------------------------------------------------------------


def cartpole_state_descriptor(obs: torch.Tensor, env) -> torch.Tensor:
    """Per-step state descriptor: (cart_pos, pole_angle) normalised to [0, 1].

    Cartpole observation layout: ``[cart_pos, cart_vel, pole_sin, pole_cos]``.
    """
    cart_pos = obs[:, 0:1]
    pole_angle = torch.atan2(obs[:, 2:3], obs[:, 3:4])
    cart_desc = (cart_pos + 2.4) / 4.8       # cart range ≈ [-2.4, 2.4]
    pole_desc = (pole_angle + 0.21) / 0.42   # angle range ≈ [-0.21, 0.21]
    return torch.cat([cart_desc, pole_desc], dim=-1).clamp(0, 1)


def cartpole_descriptor_extractor(
    transitions: QDTransition, mask: jax.Array,
) -> jax.Array:
    """Episode-level BD = mean state descriptor over non-masked timesteps."""
    valid = 1.0 - jnp.expand_dims(mask, axis=-1)
    return jnp.sum(transitions.state_desc * valid, axis=1) / jnp.sum(valid, axis=1).clip(min=1)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main() -> None:
    num_envs = args.num_envs
    episode_length = args.episode_length
    num_iterations = args.num_iterations
    seed = args.seed

    # 1. Create IsaacLab environment
    env_cfg = parse_env_cfg(
        "Isaac-Cartpole-Direct-v0",
        device=getattr(args, "device", "cuda:0") or "cuda:0",
        num_envs=num_envs,
    )
    env = gym.make("Isaac-Cartpole-Direct-v0", cfg=env_cfg).unwrapped
    print(f"[INFO] {num_envs} Cartpole envs on {env.device}")

    # 2. Wrap for QDax
    wrapper = IsaacLabQDaxWrapper(
        env=env,
        state_descriptor_fn=cartpole_state_descriptor,
        state_descriptor_length=2,
    )

    # 3. Policy network (Flax MLP)
    policy_network = MLP(
        layer_sizes=(64, 64, wrapper.action_size),
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    init_variables = jax.vmap(policy_network.init)(
        jax.random.split(subkey, num=num_envs),
        jnp.zeros((num_envs, wrapper.observation_size)),
    )

    # 4. Scoring function
    scoring_fn = make_isaaclab_scoring_fn(
        wrapper=wrapper,
        policy_fn=policy_network.apply,
        episode_length=episode_length,
        descriptor_extractor=cartpole_descriptor_extractor,
    )

    # 5. Emitter (isoline variation)
    emitter = MixingEmitter(
        mutation_fn=lambda x, r: (x, r),
        variation_fn=functools.partial(isoline_variation, iso_sigma=0.02, line_sigma=0.1),
        variation_percentage=1.0,
        batch_size=num_envs,
    )

    # 6. MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=functools.partial(default_qd_metrics, qd_offset=0.0),
    )

    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=10_000,
        num_centroids=256,
        minval=0.0,
        maxval=1.0,
        key=subkey,
    )

    # 7. Init
    key, subkey = jax.random.split(key)
    t0 = time.time()
    repertoire, emitter_state, metrics = map_elites.init(init_variables, centroids, subkey)
    print(f"[INFO] Init {time.time() - t0:.1f}s | "
          f"coverage={float(metrics['coverage']):.3f} | "
          f"max_fitness={float(metrics['max_fitness']):.1f}")

    # 8. Training loop
    for i in range(1, num_iterations + 1):
        t0 = time.time()
        key, subkey = jax.random.split(key)
        repertoire, emitter_state, metrics = map_elites.update(repertoire, emitter_state, subkey)
        dt = time.time() - t0
        if i % 10 == 0 or i == 1:
            print(f"  iter {i:4d}/{num_iterations} | {dt:.2f}s | "
                  f"cov={float(metrics['coverage']):.3f} | "
                  f"max_fit={float(metrics['max_fitness']):.1f} | "
                  f"qd={float(metrics['qd_score']):.1f}")

    # 9. Summary
    print(f"\n[DONE] best_fitness={float(jnp.max(repertoire.fitnesses)):.2f}"
          f"  coverage={float(metrics['coverage']):.3f}"
          f"  qd_score={float(metrics['qd_score']):.1f}")

    env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
