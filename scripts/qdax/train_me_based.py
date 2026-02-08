# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PGA-MAP-Elites or QDPG on any supported IsaacLab environment.

Both algorithms use the ``MAPElites`` class with CVT centroids.
They differ only in the emitter:
  * **pgame** — ``PGAMEEmitter`` (Quality-PG + GA)
  * **qdpg**  — ``QDPGEmitter``  (Quality-PG + Diversity-PG + GA)

Usage (inside the IsaacLab Docker container)::

    source scripts/qdax/setup_jax_cuda.sh

    # PGA-MAP-Elites on Ant
    ./isaaclab.sh -p scripts/qdax/train_me_based.py \\
        --algorithm pgame --task Isaac-Ant-Direct-v0 --num_envs 100

    # QDPG on Humanoid
    ./isaaclab.sh -p scripts/qdax/train_me_based.py \\
        --algorithm qdpg --task Isaac-Humanoid-Direct-v0 --num_envs 99
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/isaaclab_qdax"))

import jax
import jax.numpy as jnp

# Limit JAX GPU memory to avoid conflict with Warp/Newton
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PGA-ME / QDPG on IsaacLab envs.")
parser.add_argument("--algorithm", type=str, default="pgame", choices=["pgame", "qdpg"])
parser.add_argument("--task", type=str, default="Isaac-Ant-Direct-v0")
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--episode_length", type=int, default=200)
parser.add_argument("--num_iterations", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_centroids", type=int, default=256)
parser.add_argument("--policy_hidden", type=int, nargs="+", default=[64, 64])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -- imports after bootstrap -----------------------------------------------
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_qdax import (
    IsaacLabQDaxWrapper,
    get_env_config,
    make_isaaclab_scoring_fn,
    mean_descriptor_extractor,
)

from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import default_qd_metrics


def main() -> None:
    task = args.task
    algo = args.algorithm
    num_envs = args.num_envs
    episode_length = args.episode_length
    seed = args.seed

    # -- 1. env + wrapper --------------------------------------------------
    env_desc = get_env_config(task)
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs)
    env = gym.make(task, cfg=env_cfg).unwrapped

    wrapper = IsaacLabQDaxWrapper(
        env=env,
        state_descriptor_fn=env_desc["state_descriptor_fn"],
        state_descriptor_length=env_desc["state_descriptor_length"],
    )
    print(f"[INFO] {task} | {algo} | {num_envs} envs | "
          f"obs={wrapper.observation_size} act={wrapper.action_size} "
          f"sd={wrapper.state_descriptor_length}")

    # -- 2. policy ---------------------------------------------------------
    policy_network = MLP(
        layer_sizes=tuple(args.policy_hidden) + (wrapper.action_size,),
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    init_variables = jax.vmap(policy_network.init)(
        jax.random.split(subkey, num=num_envs),
        jnp.zeros((num_envs, wrapper.observation_size)),
    )

    # -- 3. scoring function -----------------------------------------------
    scoring_fn = make_isaaclab_scoring_fn(
        wrapper=wrapper,
        policy_fn=policy_network.apply,
        episode_length=episode_length,
        descriptor_extractor=mean_descriptor_extractor,
    )

    # -- 4. emitter --------------------------------------------------------
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=0.005, line_sigma=0.05,
    )

    if algo == "pgame":
        from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter

        emitter = PGAMEEmitter(
            config=PGAMEConfig(env_batch_size=num_envs),
            policy_network=policy_network,
            env=wrapper,
            variation_fn=variation_fn,
        )

    elif algo == "qdpg":
        from qdax.core.containers.archive import score_euclidean_novelty
        from qdax.core.emitters.dpg_emitter import DiversityPGConfig
        from qdax.core.emitters.qdpg_emitter import QDPGEmitter, QDPGEmitterConfig
        from qdax.core.emitters.qpg_emitter import QualityPGConfig

        # Split batch into 3 roughly equal parts (quality, diversity, GA).
        qpg_batch = num_envs // 3
        dpg_batch = num_envs // 3
        ga_batch = num_envs - qpg_batch - dpg_batch

        qdpg_config = QDPGEmitterConfig(
            qpg_config=QualityPGConfig(env_batch_size=qpg_batch),
            dpg_config=DiversityPGConfig(env_batch_size=dpg_batch),
            iso_sigma=0.005,
            line_sigma=0.05,
            ga_batch_size=ga_batch,
        )
        score_novelty = jax.jit(functools.partial(
            score_euclidean_novelty, num_nearest_neighb=5, scaling_ratio=1.0,
        ))
        emitter = QDPGEmitter(
            config=qdpg_config,
            policy_network=policy_network,
            env=wrapper,
            score_novelty=score_novelty,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # -- 5. MAP-Elites -----------------------------------------------------
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=functools.partial(default_qd_metrics, qd_offset=0.0),
    )

    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=wrapper.state_descriptor_length,
        num_init_cvt_samples=10_000,
        num_centroids=args.num_centroids,
        minval=0.0, maxval=1.0, key=subkey,
    )

    # -- 6. init -----------------------------------------------------------
    key, subkey = jax.random.split(key)
    t0 = time.time()
    repertoire, emitter_state, metrics = map_elites.init(
        init_variables, centroids, subkey,
    )
    print(f"[INFO] Init {time.time() - t0:.1f}s | "
          f"cov={float(metrics['coverage']):.3f} | "
          f"max_fit={float(metrics['max_fitness']):.1f}")

    # -- 7. training loop --------------------------------------------------
    for i in range(1, args.num_iterations + 1):
        t0 = time.time()
        key, subkey = jax.random.split(key)
        repertoire, emitter_state, metrics = map_elites.update(
            repertoire, emitter_state, subkey,
        )
        dt = time.time() - t0
        if i % 10 == 0 or i == 1:
            print(f"  iter {i:4d}/{args.num_iterations} | {dt:.2f}s | "
                  f"cov={float(metrics['coverage']):.3f} | "
                  f"max_fit={float(metrics['max_fitness']):.1f} | "
                  f"qd={float(metrics['qd_score']):.1f}")

    # -- 8. summary --------------------------------------------------------
    print(f"\n[DONE] best_fitness={float(jnp.max(repertoire.fitnesses)):.2f}"
          f"  coverage={float(metrics['coverage']):.3f}"
          f"  qd_score={float(metrics['qd_score']):.1f}")

    env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
