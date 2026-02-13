"""PGA-MAP-Elites training with IsaacLab environments (ask-tell mode).

This script shows how to use QDax's PGAMEEmitter with a non-Brax environment
by providing:
  1. A duck-typed env shim (IsaacLabQDEnvShim) — 3 integer properties only.
  2. A transition-collecting evaluator (JaxTransitionEvaluator) that returns
     QDTransition objects in extra_scores["transitions"].

Usage:
    isaaclab -p scripts/qd/train_pgame.py --task Isaac-Velocity-Flat-Anymal-C-v0 --num_envs 100
"""

import argparse
from isaaclab.app import AppLauncher

# ---- CLI ----
parser = argparse.ArgumentParser(description="PGA-MAP-Elites QD training for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=100, help="Number of envs (= batch size).")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--num_iterations", type=int, default=100)
parser.add_argument("--episode_length", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import functools
import time

import gymnasium as gym
import torch

import jax
import jax.numpy as jnp

from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEEmitter, PGAMEConfig
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import CSVLogger, default_qd_metrics

import logging
logging.getLogger("jax").setLevel(logging.INFO)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from pgame_utils import IsaacLabQDEnvShim, JaxTransitionEvaluator

# ---------- Hyperparameters ----------
# MAP-Elites / variation
ISO_SIGMA = 0.005
LINE_SIGMA = 0.05
NUM_INIT_CVT_SAMPLES = 50000
NUM_CENTROIDS = 1024
NUM_DESCRIPTORS = 2
MIN_DESCRIPTOR = 0.0
MAX_DESCRIPTOR = 1.0
POLICY_HIDDEN_SIZE = 64

# PGA-ME specific
PROPORTION_MUTATION_GA = 0.5
NUM_CRITIC_TRAINING_STEPS = 300
NUM_PG_TRAINING_STEPS = 100
REPLAY_BUFFER_SIZE = 1_000_000
CRITIC_HIDDEN_LAYER_SIZE = (256, 256)
CRITIC_LR = 3e-4
GREEDY_LR = 3e-4
POLICY_LR = 1e-3
DISCOUNT = 0.99
REWARD_SCALING = 1.0
BATCH_SIZE_PG = 256
SOFT_TAU = 0.005
POLICY_DELAY = 2
NOISE_CLIP = 0.5
POLICY_NOISE = 0.2


def main():
    device = args_cli.device
    batch_size = args_cli.num_envs
    num_iterations = args_cli.num_iterations
    episode_length = args_cli.episode_length
    seed = args_cli.seed

    # ---- Create IsaacLab environment ----
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=batch_size,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    obs_space = env.observation_space
    obs_dim = (
        obs_space["policy"].shape[-1]
        if isinstance(obs_space, gym.spaces.Dict)
        else obs_space.shape[-1]
    )
    action_dim = env.action_space.shape[-1]
    print(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}, batch_size={batch_size}")

    # ---- QDax Flax MLP policy ----
    policy_layer_sizes = (POLICY_HIDDEN_SIZE, POLICY_HIDDEN_SIZE, action_dim)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # ---- Env shim — duck-typed replacement for QDEnv ----
    env_shim = IsaacLabQDEnvShim(
        observation_size=obs_dim,
        action_size=action_dim,
        state_descriptor_length=NUM_DESCRIPTORS,
    )

    # ---- Transition-collecting evaluator ----
    evaluator = JaxTransitionEvaluator(
        env=env,
        num_envs=batch_size,
        policy_network=policy_network,
        num_steps=episode_length,
        # state_descriptor_fn=None uses default obs[:, :2]
    )

    # ---- Bridge function ----
    def evaluate_genotypes(genotypes_jax):
        fitnesses_torch, descriptors_torch, extra = evaluator.evaluate(genotypes_jax)
        fitnesses_jax = jnp.from_dlpack(fitnesses_torch)
        descriptors_jax = jnp.from_dlpack(descriptors_torch.contiguous())
        return fitnesses_jax, descriptors_jax, extra  # extra has "transitions"

    # ---- Initial population ----
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, obs_dim))
    init_genotypes = jax.vmap(policy_network.init)(keys, fake_batch)

    num_params = sum(x.size for x in jax.tree.leaves(init_genotypes)) // batch_size
    print(f"[INFO] Policy parameters per individual: {num_params}")

    # ---- PGA-ME Emitter ----
    pgame_config = PGAMEConfig(
        env_batch_size=batch_size,
        proportion_mutation_ga=PROPORTION_MUTATION_GA,
        num_critic_training_steps=NUM_CRITIC_TRAINING_STEPS,
        num_pg_training_steps=NUM_PG_TRAINING_STEPS,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        critic_hidden_layer_size=CRITIC_HIDDEN_LAYER_SIZE,
        critic_learning_rate=CRITIC_LR,
        greedy_learning_rate=GREEDY_LR,
        policy_learning_rate=POLICY_LR,
        noise_clip=NOISE_CLIP,
        policy_noise=POLICY_NOISE,
        discount=DISCOUNT,
        reward_scaling=REWARD_SCALING,
        batch_size=BATCH_SIZE_PG,
        soft_tau_update=SOFT_TAU,
        policy_delay=POLICY_DELAY,
    )

    variation_fn = functools.partial(
        isoline_variation, iso_sigma=ISO_SIGMA, line_sigma=LINE_SIGMA
    )

    pgame_emitter = PGAMEEmitter(
        config=pgame_config,
        policy_network=policy_network,
        env=env_shim,  # duck-typed shim — NOT a real Brax env
        variation_fn=variation_fn,
    )

    # ---- MAP-Elites ----
    metrics_function = functools.partial(default_qd_metrics, qd_offset=0.0)

    map_elites = MAPElites(
        scoring_function=None,
        emitter=pgame_emitter,
        metrics_function=metrics_function,
    )

    # CVT centroids
    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=NUM_DESCRIPTORS,
        num_init_cvt_samples=NUM_INIT_CVT_SAMPLES,
        num_centroids=NUM_CENTROIDS,
        minval=MIN_DESCRIPTOR,
        maxval=MAX_DESCRIPTOR,
        key=subkey,
    )

    # ---- Evaluate initial population ----
    print("[INFO] Evaluating initial population...")
    fitnesses, descriptors, extra_scores = evaluate_genotypes(init_genotypes)

    key, subkey = jax.random.split(key)
    repertoire, emitter_state, _ = map_elites.init_ask_tell(
        genotypes=init_genotypes,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        key=subkey,
        extra_scores=extra_scores,
    )

    # ---- Ask-Tell loop ----
    ask_fn = jax.jit(map_elites.ask)
    tell_fn = jax.jit(map_elites.tell)

    log_metrics = dict.fromkeys(
        ["iteration", "qd_score", "coverage", "max_fitness", "time"],
        jnp.array([]),
    )
    csv_logger = CSVLogger("pgame-logs.csv", header=list(log_metrics.keys()))

    print(f"[INFO] Starting PGA-MAP-Elites for {num_iterations} iterations")
    global_start_time = time.perf_counter()
    for i in range(num_iterations):
        start_time = time.perf_counter()

        # ASK
        key, subkey = jax.random.split(key)
        genotypes, extra_info = ask_fn(repertoire, emitter_state, subkey)
        ask_time = time.perf_counter() - start_time

        # EVALUATE (returns transitions in extra_scores for PGA-ME)
        fitnesses, descriptors, extra_scores = evaluate_genotypes(genotypes)
        eval_time = time.perf_counter() - start_time - ask_time

        # TELL (trains critic/actor internally via replay buffer)
        repertoire, emitter_state, current_metrics = tell_fn(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            repertoire=repertoire,
            emitter_state=emitter_state,
            extra_scores=extra_scores,
            extra_info=extra_info,
        )

        elapsed = time.perf_counter() - start_time
        tell_time = elapsed - ask_time - eval_time

        current_metrics["iteration"] = i
        current_metrics["time"] = elapsed
        current_metrics = jax.tree.map(lambda x: jnp.array([x]), current_metrics)
        log_metrics = jax.tree.map(
            lambda old, new: jnp.concatenate([old, new], axis=0),
            log_metrics,
            current_metrics,
        )
        csv_logger.log(jax.tree.map(lambda x: x[-1], log_metrics))

        print(
            f"  Iter {i:4d} | "
            f"max_fitness={float(current_metrics['max_fitness'][0]):.2f} | "
            f"coverage={float(current_metrics['coverage'][0]):.4f} | "
            f"qd_score={float(current_metrics['qd_score'][0]):.2f} | "
            f"time={elapsed:.2f}s (ask={ask_time:.2f}s, eval={eval_time:.2f}s, tell={tell_time:.2f}s)"
        )

    global_elapsed = time.perf_counter() - global_start_time
    print(f"[INFO] PGA-MAP-Elites completed {num_iterations} iters in {global_elapsed:.2f}s.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
