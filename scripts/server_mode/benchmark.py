import torch
from socket_env.socket_env_client import SocketEnvClient
import time

def main(env: SocketEnvClient, task: str, num_envs: int, steps: int, benchmark_interval: int):
    
    env.make(task, num_envs)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, extras = env.reset()

    start = time.perf_counter()
    # simulate environment
    for step in range(steps):
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.device) - 1
            # apply actions
            obs, rewards, terminated, truncated, _ = env.step(actions)
        
        if step % benchmark_interval == 0:
            print(f"[INFO]: Steps/s = {benchmark_interval / (time.perf_counter() - start):.2f}")
            start = time.perf_counter()

    # close the simulator
    env.close()

if __name__ == "__main__":
    env = SocketEnvClient()
    main(env, task="Isaac-Cartpole-Direct-v0", num_envs=2048, steps=10000, benchmark_interval=100)
    env.stop()
    