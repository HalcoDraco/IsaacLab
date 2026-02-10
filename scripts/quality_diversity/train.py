"""Quality-Diversity training with IsaacLab environments.

Supported algorithms
--------------------
* **MAP-Elites** — standard mutation-based QD (pyribs ``GaussianEmitter``).
* **PGA-MAP-Elites** — policy-gradient-assisted QD (custom ``PGAEmitter``
  backed by an SB3 off-policy critic + ``GaussianEmitter`` for the GA half).

Usage (inside the IsaacLab Docker container)
--------------------------------------------
.. code-block:: bash

    # MAP-Elites on Cartpole
    isaaclab -p scripts/quality_diversity/train.py \\
        --task Isaac-Cartpole-Direct-v0 --num_envs 128 --headless \\
        --algo map_elites --generations 50

    # PGA-MAP-Elites on Cartpole (TD3 critic)
    isaaclab -p scripts/quality_diversity/train.py \\
        --task Isaac-Cartpole-Direct-v0 --num_envs 128 --headless \\
        --algo pga_map_elites --generations 50 --rl_algorithm TD3

    # PGA-MAP-Elites with SAC critic
    isaaclab -p scripts/quality_diversity/train.py \\
        --task Isaac-Ant-Direct-v0 --num_envs 256 --headless \\
        --algo pga_map_elites --rl_algorithm SAC --generations 200
"""

from __future__ import annotations

# ── IsaacLab app launcher (MUST come before any other Isaac imports) ──────
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="QD training with pyribs + IsaacLab")

# Environment / sim args.
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)

# Algorithm args.
parser.add_argument("--algo", type=str, default="map_elites",
                    choices=["map_elites", "pga_map_elites"])
parser.add_argument("--generations", type=int, default=100)
parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64],
                    help="Hidden layer sizes for the policy MLP.")
parser.add_argument("--sigma", type=float, default=0.02,
                    help="Mutation std for GaussianEmitter.")
parser.add_argument("--max_steps", type=int, default=500,
                    help="Max simulation steps per episode.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--save_path", type=str, default=None,
                    help="Path to save the final archive (pickle).")

# Measure configuration.
parser.add_argument("--measure", type=str, default="obs_slice",
                    choices=["obs_slice", "final_xy", "mean_xy_vel", "cartpole"],
                    help="Behavioral descriptor type.")
parser.add_argument("--measure_indices", type=int, nargs="+", default=[0, 2],
                    help="Observation indices for obs_slice measure.")
parser.add_argument("--measure_low", type=float, nargs="+", default=[-3.0, -3.0],
                    help="Lower bounds for the measure grid.")
parser.add_argument("--measure_high", type=float, nargs="+", default=[3.0, 3.0],
                    help="Upper bounds for the measure grid.")
parser.add_argument("--grid_dims", type=int, nargs="+", default=[50, 50],
                    help="Number of cells per measure dimension.")

# PGA-specific args.
parser.add_argument("--pg_steps", type=int, default=10)
parser.add_argument("--critic_updates", type=int, default=300)
parser.add_argument("--pga_proportion", type=float, default=0.5,
                    help="Fraction of num_envs used by the PGA emitter (rest for GA).")
parser.add_argument("--rl_algorithm", type=str, default="TD3",
                    choices=["TD3", "SAC"],
                    help="SB3 off-policy algorithm for PGA critic training.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True  # always headless for QD

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Regular imports (after sim is booted) ─────────────────────────────────
import pickle
import time

import gymnasium as gym
import numpy as np

import isaaclab_tasks  # noqa: F401 – registers tasks
from isaaclab.utils import close_simulation
from isaaclab_tasks.utils import parse_env_cfg

# Local modules — make sure the script directory is on the path.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from emitters import PGAEmitter
from evaluator import Evaluator
from measures import (
    CartpoleMeasure,
    FinalXYPosition,
    MeanXYVelocity,
    ObservationSlice,
)
from policy import MLPPolicy, flat_params

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.schedulers import Scheduler


# ── Helpers ───────────────────────────────────────────────────────────────

def build_measure_fn(args):
    """Construct the measure function from CLI args."""
    if args.measure == "obs_slice":
        return ObservationSlice(args.measure_indices)
    elif args.measure == "final_xy":
        return FinalXYPosition()
    elif args.measure == "mean_xy_vel":
        return MeanXYVelocity()
    elif args.measure == "cartpole":
        return CartpoleMeasure()
    else:
        raise ValueError(f"Unknown measure: {args.measure}")


def build_archive(param_dim: int, args) -> GridArchive:
    """Build a pyribs GridArchive."""
    ranges = list(zip(args.measure_low, args.measure_high))
    return GridArchive(
        solution_dim=param_dim,
        dims=args.grid_dims,
        ranges=ranges,
        seed=args.seed,
    )


def build_scheduler(archive, obs_dim, act_dim, args, device_str):
    """Build the pyribs Scheduler for the chosen algorithm.

    Returns ``(scheduler, pga_emitter_or_None)``.
    """
    hidden = tuple(args.hidden)
    x0 = flat_params(MLPPolicy(obs_dim, act_dim, hidden)).cpu().numpy()  # (param_dim,)

    if args.algo == "map_elites":
        emitters = [
            GaussianEmitter(
                archive, sigma=args.sigma, x0=x0,
                batch_size=args.num_envs, seed=args.seed,
            )
        ]
        return Scheduler(archive, emitters), None

    elif args.algo == "pga_map_elites":
        pga_batch = max(1, int(args.num_envs * args.pga_proportion))
        ga_batch = args.num_envs - pga_batch
        assert ga_batch > 0, (
            f"pga_proportion={args.pga_proportion} leaves 0 envs for the GA "
            f"emitter (num_envs={args.num_envs})."
        )

        pga = PGAEmitter(
            archive,
            obs_dim=obs_dim,
            action_dim=act_dim,
            policy_hidden=hidden,
            batch_size=pga_batch,
            pg_steps=args.pg_steps,
            rl_algorithm=args.rl_algorithm,
            critic_updates=args.critic_updates,
            device=device_str,
            seed=args.seed,
        )
        ga = GaussianEmitter(
            archive, sigma=args.sigma, x0=x0,
            batch_size=ga_batch, seed=args.seed + 1,
        )
        return Scheduler(archive, [pga, ga]), pga

    else:
        raise ValueError(f"Unknown algo: {args.algo}")


# ── Transition callback (for PGA critic) ──────────────────────────────────

def _make_transition_cb(pga_emitter):
    """Return a callback for the evaluator, or None if not using PGA."""
    if pga_emitter is None:
        return None
    return pga_emitter.add_transitions


# ── Main training loop ───────────────────────────────────────────────────

def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    hidden = tuple(args.hidden)
    device_str = str(env.unwrapped.device)

    print(f"Task: {args.task}  |  obs={obs_dim}  act={act_dim}  "
          f"num_envs={args.num_envs}  algo={args.algo}")

    # Measure.
    measure_fn = build_measure_fn(args)
    measure_dim = measure_fn.dim
    assert len(args.measure_low) == measure_dim, (
        f"measure_low has {len(args.measure_low)} dims, expected {measure_dim}")
    assert len(args.grid_dims) == measure_dim, (
        f"grid_dims has {len(args.grid_dims)} dims, expected {measure_dim}")

    # Evaluator.
    evaluator = Evaluator(
        env, obs_dim, act_dim, hidden, measure_fn, max_steps=args.max_steps,
    )
    param_dim = evaluator.param_dim
    print(f"Policy params: {param_dim}  |  Measure dim: {measure_dim}")

    # Archive + scheduler.
    archive = build_archive(param_dim, args)
    scheduler, pga_emitter = build_scheduler(
        archive, obs_dim, act_dim, args, device_str,
    )
    transition_cb = _make_transition_cb(pga_emitter)

    # ── Generation loop ───────────────────────────────────────────────────
    for gen in range(1, args.generations + 1):
        t_start = time.perf_counter()

        solutions = scheduler.ask()
        t_ask = time.perf_counter()

        objectives, measures = evaluator.evaluate(
            solutions, transition_cb=transition_cb,
        )
        t_eval = time.perf_counter()

        scheduler.tell(objectives, measures)
        t_tell = time.perf_counter()

        if gen % args.log_interval == 0:
            stats = archive.stats
            elapsed = t_tell - t_start
            print(
                f"Gen {gen:>4d}  |  "
                f"{elapsed:5.1f}s  "
                f"(ask {t_ask - t_start:.2f}  "
                f"eval {t_eval - t_ask:.2f}  "
                f"tell {t_tell - t_eval:.2f})  |  "
                f"archive: {len(archive):>5d}  |  "
                f"coverage: {len(archive) / archive.cells * 100:5.1f}%  |  "
                f"best: {stats.obj_max:+8.2f}  |  "
                f"mean: {stats.obj_mean:+8.2f}"
            )

    # ── Save ──────────────────────────────────────────────────────────────
    if args.save_path:
        with open(args.save_path, "wb") as f:
            pickle.dump(archive, f)
        print(f"Archive saved to {args.save_path}")

    env.close()
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_simulation(simulation_app)
