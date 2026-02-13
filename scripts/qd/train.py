import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="MAP-Elites QD training for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments (= MAP-Elites batch size).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_iterations", type=int, default=100, help="Number of MAP-Elites iterations.")
parser.add_argument("--episode_length", type=int, default=300, help="Max steps per evaluation episode.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
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
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import CSVLogger, default_qd_metrics

import logging
logging.getLogger("jax").setLevel(logging.INFO)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from evaluator import JaxEvaluator

# ---------- MAP-Elites hyperparameters ----------
ISO_SIGMA = 0.005
LINE_SIGMA = 0.05
NUM_INIT_CVT_SAMPLES = 50000
NUM_CENTROIDS = 1024
NUM_DESCRIPTORS = 2
MIN_DESCRIPTOR = 0.0
MAX_DESCRIPTOR = 1.0
POLICY_HIDDEN_SIZE = 64


class SimplePolicy(torch.nn.Module):
    """Simple MLP policy with tanh output (bounded actions in [-1, 1])."""

    def __init__(self, obs_dim: int, hidden: int, action_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, action_dim),
            torch.nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    device = args_cli.device
    batch_size = args_cli.num_envs  # one policy per environment
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

    # Infer observation and action dimensions
    obs_space = env.observation_space
    obs_dim = (
        obs_space["policy"].shape[-1]
        if isinstance(obs_space, gym.spaces.Dict)
        else obs_space.shape[-1]
    )
    action_dim = env.action_space.shape[-1]
    print(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}, batch_size={batch_size}")

    # ---- Create QDax Flax MLP policy ----
    policy_layer_sizes = (POLICY_HIDDEN_SIZE, POLICY_HIDDEN_SIZE, action_dim)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # ---- JAX Evaluator (Flax policy, DLPack bridge for IsaacLab) ----
    evaluator = JaxEvaluator(
        env=env,
        num_envs=batch_size,
        policy_network=policy_network,
        num_steps=episode_length,
    )

    # ---- Bridge function: evaluate JAX genotypes in IsaacLab ----
    def evaluate_genotypes(genotypes_jax):
        """Evaluate JAX PyTree genotypes in IsaacLab, return JAX arrays."""
        fitnesses_torch, descriptors_torch = evaluator.evaluate(genotypes_jax)
        fitnesses_jax = jnp.from_dlpack(fitnesses_torch)
        descriptors_jax = jnp.from_dlpack(descriptors_torch.contiguous())
        return fitnesses_jax, descriptors_jax, {}

    # ---- QDax MAP-Elites setup ----
    key = jax.random.key(seed)

    # Initial population: random Flax network parameters (JAX PyTree genotypes)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, obs_dim))
    init_genotypes = jax.vmap(policy_network.init)(keys, fake_batch)

    num_params = sum(x.size for x in jax.tree.leaves(init_genotypes)) // batch_size
    print(f"[INFO] Policy parameters per individual: {num_params}")

    # Emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=ISO_SIGMA, line_sigma=LINE_SIGMA
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    # Metrics
    metrics_function = functools.partial(default_qd_metrics, qd_offset=0.0)

    # MAP-Elites instance (no scoring_function — using ask-tell)
    map_elites = MAPElites(
        scoring_function=None,
        emitter=mixing_emitter,
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

    # Initialize repertoire and emitter state
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, _ = map_elites.init_ask_tell(
        genotypes=init_genotypes,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        key=subkey,
        extra_scores=extra_scores,
    )

    # ---- MAP-Elites ask-tell loop ----
    ask_fn = jax.jit(map_elites.ask)
    tell_fn = jax.jit(map_elites.tell)

    log_metrics = dict.fromkeys(
        ["iteration", "qd_score", "coverage", "max_fitness", "time"],
        jnp.array([]),
    )
    csv_logger = CSVLogger("mapelites-logs.csv", header=list(log_metrics.keys()))

    print(f"[INFO] Starting MAP-Elites for {num_iterations} iterations")
    global_start_time = time.perf_counter()
    for i in range(num_iterations):
        start_time = time.perf_counter()

        # ASK: generate candidate genotypes (JAX, JIT-compiled)
        key, subkey = jax.random.split(key)
        genotypes, extra_info = ask_fn(repertoire, emitter_state, subkey)

        ask_time = time.perf_counter() - start_time

        # EVALUATE: run candidates in IsaacLab (PyTorch, not JIT)
        fitnesses, descriptors, extra_scores = evaluate_genotypes(genotypes)

        eval_time = time.perf_counter() - start_time - ask_time

        # TELL: update the repertoire (JAX, JIT-compiled)
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

        # Log metrics
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
    print(f"[INFO] MAP-Elites completed {num_iterations} iterations in {global_elapsed:.2f}s.")
    print("[INFO] Training complete.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
