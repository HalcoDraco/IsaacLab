"""Isaac Lab GPU IPC Environment Server.

Runs **inside the Docker container**. Creates an Isaac Lab task and exposes it
to a host-side training algorithm via the GPU IPC bridge (zero-copy shared CUDA
buffers + TCP synchronisation).

Usage (inside the container)::

    # Headless, 4096 envs, default port 5555
    python scripts/qd/isaac_env_server.py --task Isaac-Velocity-Rough-Anymal-C-v0 \\
        --num_envs 4096 --headless

    # Custom port
    python scripts/qd/isaac_env_server.py --task Isaac-Reach-Franka-v0 \\
        --num_envs 512 --headless --port 6000

The server waits for exactly one client to connect, then enters the step loop.
It shuts down when the client disconnects or sends a close message.
"""

# ----- Isaac Sim must be launched BEFORE any Omniverse / Isaac Lab imports -----

import argparse
import logging
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Isaac Lab GPU IPC Environment Server")
parser.add_argument("--task", type=str, required=True, help="Registered Isaac Lab task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--port", type=int, default=5555, help="TCP port for the IPC bridge.")
parser.add_argument("--host", type=str, default="0.0.0.0", help="TCP bind address.")
# AppLauncher adds --headless, --device, etc.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Boot the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----- Safe to import everything else now -----

import gymnasium as gym

import isaaclab_tasks  # noqa: F401  – registers all task entry points

from isaaclab_tasks.utils import parse_env_cfg

# Add the script's own directory to sys.path so we can import the bridge module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_ipc_bridge import IsaacEnvServer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("isaac_env_server")


def main() -> None:
    # Build env config (respects --device, --num_envs, etc.)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )

    # Create the vectorised Isaac Lab environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    logger.info(f"Created env '{args_cli.task}' — obs_space={env.observation_space}, act_space={env.action_space}")

    # Create and start the GPU IPC server (blocks until client disconnects)
    server = IsaacEnvServer(env, host=args_cli.host, port=args_cli.port)
    server.serve()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
