# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PGA-AURORA on any supported IsaacLab environment.

AURORA replaces hand-crafted behavior descriptors with a learned latent
encoding (LSTM seq2seq autoencoder).  Uses an ``UnstructuredRepertoire``
instead of CVT centroids, with periodic AE retraining and container-size
control.

Usage (inside the IsaacLab Docker container)::

    source scripts/qdax/setup_jax_cuda.sh
    ./isaaclab.sh -p scripts/qdax/train_pga_aurora.py \\
        --task Isaac-Ant-Direct-v0 --num_envs 100
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/isaaclab_qdax"))

import jax
import jax.numpy as jnp

# Limit JAX GPU memory to avoid conflict with Warp/Newton
import os as _os
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PGA-AURORA on IsaacLab envs.")
parser.add_argument("--task", type=str, default="Isaac-Ant-Direct-v0")
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--episode_length", type=int, default=200)
parser.add_argument("--num_iterations", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--hidden_size", type=int, default=5,
                    help="LSTM latent dim = learned descriptor dim.")
parser.add_argument("--traj_sampling_freq", type=int, default=10)
parser.add_argument("--max_obs_size", type=int, default=25)
parser.add_argument("--l_value_init", type=float, default=0.2)
parser.add_argument("--n_target", type=int, default=1024,
                    help="Target number of individuals in repertoire.")
parser.add_argument("--policy_hidden", type=int, nargs="+", default=[64, 64])
parser.add_argument("--log_freq", type=int, default=5)
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
    make_isaaclab_aurora_scoring_fn,
    mean_descriptor_extractor,
)

from qdax.core.aurora import AURORA
from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import AuroraExtraInfoNormalization
from qdax.tasks.brax.descriptor_extractors import get_aurora_encoding
from qdax.utils import train_seq2seq


def main() -> None:
    task = args.task
    num_envs = args.num_envs
    episode_length = args.episode_length
    seed = args.seed
    hidden_size = args.hidden_size
    traj_freq = args.traj_sampling_freq
    max_obs = args.max_obs_size
    log_freq = args.log_freq
    observations_key = "observations"

    # -- 1. Pre-compute dims & init LSTM on CPU ----------------------------
    # The LSTM weight init uses QR decomposition (cusolver), which conflicts
    # with Warp/Newton's CUDA context. We force LSTM init onto CPU and move
    # the params to GPU afterward.
    env_desc = get_env_config(task)
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs)
    lstm_input_dim = max_obs
    observations_dims = (episode_length // traj_freq, lstm_input_dim)

    model = train_seq2seq.get_model(
        observations_dims[-1], True, hidden_size=hidden_size,
    )
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    # Force init on CPU to avoid cusolver conflict with Warp
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        model_params = train_seq2seq.get_initial_params(
            model, subkey, (1, *observations_dims),
        )
    # Transfer to GPU
    model_params = jax.device_put(model_params, jax.devices("gpu")[0])
    print("[INFO] LSTM autoencoder params initialized (on CPU, moved to GPU).")

    # -- 2. env + wrapper --------------------------------------------------
    env = gym.make(task, cfg=env_cfg).unwrapped

    wrapper = IsaacLabQDaxWrapper(
        env=env,
        state_descriptor_fn=env_desc["state_descriptor_fn"],
        state_descriptor_length=env_desc["state_descriptor_length"],
    )
    print(f"[INFO] {task} | PGA-AURORA | {num_envs} envs | "
          f"obs={wrapper.observation_size} act={wrapper.action_size}")

    # Update obs_dim now that we know the true value
    obs_dim = min(wrapper.observation_size, max_obs)
    # If obs_dim < max_obs, the LSTM was created with a larger dim than needed.
    # Recreate if there's a mismatch (only the dim matters for the model).
    if obs_dim != lstm_input_dim:
        observations_dims = (episode_length // traj_freq, obs_dim)
        model = train_seq2seq.get_model(
            observations_dims[-1], True, hidden_size=hidden_size,
        )
        # Re-init on CPU to avoid cusolver conflict
        key, subkey = jax.random.split(key)
        with jax.default_device(jax.devices("cpu")[0]):
            model_params = train_seq2seq.get_initial_params(
                model, subkey, (1, *observations_dims),
            )
        # Move to GPU
        model_params = jax.device_put(model_params, jax.devices("gpu")[0])

    # -- 3. policy ---------------------------------------------------------
    policy_network = MLP(
        layer_sizes=tuple(args.policy_hidden) + (wrapper.action_size,),
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    key, subkey = jax.random.split(key)
    init_variables = jax.vmap(policy_network.init)(
        jax.random.split(subkey, num=num_envs),
        jnp.zeros((num_envs, wrapper.observation_size)),
    )

    # -- 4. AURORA scoring function ----------------------------------------
    scoring_fn = make_isaaclab_aurora_scoring_fn(
        wrapper=wrapper,
        policy_fn=policy_network.apply,
        episode_length=episode_length,
        descriptor_extractor=mean_descriptor_extractor,
        traj_sampling_freq=traj_freq,
        max_obs_size=obs_dim,
        observations_key=observations_key,
    )

    encoder_fn = jax.jit(functools.partial(get_aurora_encoding, model=model))
    train_fn = functools.partial(
        train_seq2seq.lstm_ae_train, model=model, batch_size=128,
    )

    # -- 5. emitter (PGA-ME style) ----------------------------------------
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=0.005, line_sigma=0.05,
    )
    pg_emitter = PGAMEEmitter(
        config=PGAMEConfig(env_batch_size=num_envs),
        policy_network=policy_network,
        env=wrapper,
        variation_fn=variation_fn,
    )

    # -- 6. metrics --------------------------------------------------------
    def metrics_fn(repertoire: UnstructuredRepertoire) -> Dict:
        valid = repertoire.fitnesses != -jnp.inf
        return {
            "qd_score": jnp.sum(repertoire.fitnesses, where=valid),
            "max_fitness": jnp.max(repertoire.fitnesses),
            "coverage": 100.0 * jnp.mean(valid.astype(jnp.float32)),
        }

    # -- 7. AURORA algorithm -----------------------------------------------
    aurora = AURORA(
        scoring_function=scoring_fn,
        emitter=pg_emitter,
        metrics_function=metrics_fn,
        encoder_function=encoder_fn,
        training_function=train_fn,
        observations_key=observations_key,
    )

    aurora_extra_info = AuroraExtraInfoNormalization.create(
        model_params,
        mean_observations=jnp.zeros(observations_dims[-1]),
        std_observations=jnp.ones(observations_dims[-1]),
    )

    # -- 8. init -----------------------------------------------------------
    # NOTE: aurora.init() already calls self.train() internally
    max_size = 2 * args.n_target  # room for the unstructured repertoire
    key, subkey = jax.random.split(key)
    t0 = time.time()
    repertoire, emitter_state, metrics, aurora_extra_info = aurora.init(
        init_variables,
        aurora_extra_info,
        jnp.asarray(args.l_value_init),
        max_size,
        subkey,
    )
    print(f"[INFO] Init {time.time() - t0:.1f}s | "
          f"cov={float(metrics['coverage']):.1f}% | "
          f"max_fit={float(metrics['max_fitness']):.1f}")

    # -- 9. training loop --------------------------------------------------
    # Schedule for AE retraining (increasingly spaced)
    default_update_base = 10
    update_base = max(1, default_update_base // log_freq)
    schedules = set(jnp.cumsum(jnp.arange(update_base, 1000, update_base)).tolist())

    previous_error = jnp.sum(repertoire.fitnesses != -jnp.inf) - args.n_target
    container_size_control_fn = jax.jit(aurora.container_size_control)

    for iteration in range(args.num_iterations):
        t0 = time.time()

        # Run log_freq update steps
        for _ in range(log_freq):
            key, subkey = jax.random.split(key)
            repertoire, emitter_state, metrics = aurora.update(
                repertoire, emitter_state, subkey,
                aurora_extra_info=aurora_extra_info,
            )

        dt = time.time() - t0

        # Periodic AE retraining or container size control
        if (iteration + 1) in schedules:
            key, subkey = jax.random.split(key)
            repertoire, aurora_extra_info = aurora.train(
                repertoire, model_params, iteration, subkey,
            )
        elif iteration % 2 == 0:
            repertoire, previous_error = container_size_control_fn(
                repertoire, target_size=args.n_target,
                previous_error=previous_error,
            )

        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"  block {iteration + 1:4d}/{args.num_iterations} "
                  f"({log_freq} iters) | {dt:.2f}s | "
                  f"cov={float(metrics['coverage']):.1f}% | "
                  f"max_fit={float(metrics['max_fitness']):.1f} | "
                  f"qd={float(metrics['qd_score']):.1f}")

    # -- 10. summary -------------------------------------------------------
    print(f"\n[DONE] best_fitness={float(jnp.max(repertoire.fitnesses)):.2f}"
          f"  coverage={float(metrics['coverage']):.1f}%"
          f"  qd_score={float(metrics['qd_score']):.1f}")

    env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
