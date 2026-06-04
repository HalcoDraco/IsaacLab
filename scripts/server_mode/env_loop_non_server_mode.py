import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Launches the server environment")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--task", type=str, default="Isaac-Repose-Cube-Shadow-Direct-v0", help="Task to run the environment loop on."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to run in parallel."
)
parser.add_argument(
    "--result_file",
    type=str,
    default=None,
    help="Optional file path where the total execution time is written.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


import time

task = args_cli.task
num_envs = args_cli.num_envs
steps = 10000
disable_fabric = args_cli.disable_fabric
result_file = args_cli.result_file

def env_loop(task: str, num_envs: int, steps: int, benchmark_interval: int | None):
    
    device = "cuda"

    env_cfg = parse_env_cfg(
        task, 
        device=device, 
        num_envs=num_envs, 
        use_fabric=not disable_fabric
    )

    try:
        env = gym.make(task, cfg=env_cfg, render_mode=None)

        # print info (this is vectorized environment)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment
        obs, extras = env.reset()

        # 100 steps warmup
        for _ in range(100):
            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=device) - 1
                obs, rewards, terminated, truncated, extras = env.step(actions)

        global_start = time.perf_counter()
        start = time.perf_counter()
        # simulate environment
        for step in range(steps):
            # run everything in inference mode
            with torch.inference_mode():
                # sample actions from -1 to 1
                actions = 2 * torch.rand(env.action_space.shape, device=device) - 1
                # apply actions
                obs, rewards, terminated, truncated, extras = env.step(actions)
            
            if benchmark_interval and step % benchmark_interval == 0:
                print(f"[INFO]: Steps/s = {benchmark_interval / (time.perf_counter() - start):.2f}")
                start = time.perf_counter()
        
        total_time = time.perf_counter() - global_start

    except Exception as e:
        print(f"Error running task {task} with {num_envs} envs: {e}")
        total_time = -1
    finally:
        # close the simulator
        if 'env' in locals():
            env.close()
            del env

    return total_time

if __name__ == "__main__":
    benchmark_interval = None
    total_time = env_loop(task, num_envs, steps, benchmark_interval)
    if result_file is not None:
        Path(result_file).write_text(f"{total_time}\n", encoding="utf-8")
    else:
        print(total_time)
    simulation_app.close()
    
    
    