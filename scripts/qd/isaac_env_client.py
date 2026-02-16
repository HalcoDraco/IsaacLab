#!/usr/bin/env python3
"""Isaac Lab GPU IPC Environment Client — demo / smoke-test script.

Runs **on the host** (outside Docker). Connects to an ``isaac_env_server.py``
running inside the container, receives observations through zero-copy shared
CUDA memory, sends random actions back, and prints basic statistics.

Usage::

    # After starting the server inside the container:
    python scripts/qd/isaac_env_client.py --port 5555 --num_steps 1000

For integration with your own algorithm, import the client class directly::

    from gpu_ipc_bridge import IsaacEnvClient

    client = IsaacEnvClient(port=5555)
    obs = client.reset()                       # (num_envs, obs_size) on GPU
    obs, rew, done, trunc, info = client.step(action)
    client.close()
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

# Ensure the bridge module is importable from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_ipc_bridge import IsaacEnvClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("isaac_env_client")


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Lab GPU IPC Client — random-action demo")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server address.")
    parser.add_argument("--port", type=int, default=5555, help="Server TCP port.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of env steps to run.")
    args = parser.parse_args()

    client = IsaacEnvClient(host=args.host, port=args.port)

    # Connect and get initial observation
    obs = client.reset()
    logger.info(
        f"Connected — num_envs={client.num_envs}, obs_size={client.obs_size}, "
        f"act_size={client.act_size}, device={obs.device}"
    )

    total_reward = torch.zeros(client.num_envs, device=obs.device)
    t0 = time.perf_counter()
    log_interval = max(1, args.num_steps // 10)

    for step in range(1, args.num_steps + 1):
        # Random actions in [-1, 1]
        action = 2.0 * torch.rand(client.num_envs, client.act_size, device=obs.device) - 1.0

        obs, reward, done, truncated, info = client.step(action)
        total_reward += reward

        if step % log_interval == 0:
            elapsed = time.perf_counter() - t0
            fps = log_interval / elapsed
            logger.info(
                f"Step {step}/{args.num_steps} — "
                f"steps/s: {fps:.1f}, "
                f"mean_reward: {reward.mean():.4f}, "
                f"mean_total: {total_reward.mean():.4f}, "
                f"done_frac: {done.mean():.3f}"
            )
            t0 = time.perf_counter()

    logger.info(f"Finished {args.num_steps} steps — final mean total reward: {total_reward.mean():.4f}")
    client.close()


if __name__ == "__main__":
    main()
