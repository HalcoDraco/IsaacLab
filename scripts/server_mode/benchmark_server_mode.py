import torch
from isaaclab_fork import SocketEnvClient
import time

def env_loop(task: str, num_envs: int, steps: int, benchmark_interval: int | None):
    
    env = SocketEnvClient()

    try:
        env.make(task, num_envs)

        # print info (this is vectorized environment)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment
        obs, extras = env.reset()

        # 100 steps warmup
        for _ in range(100):
            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=env.device) - 1
                obs, rewards, terminated, truncated, extras = env.step(actions)

        global_start = time.perf_counter()
        start = time.perf_counter()
        # simulate environment
        for step in range(steps):
            # run everything in inference mode
            with torch.inference_mode():
                # sample actions from -1 to 1
                actions = 2 * torch.rand(env.action_space.shape, device=env.device) - 1
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
        env.close()
        env.disconnect()
        del env

    return total_time

def benchmark_task(task: str, num_envs_pow2: tuple[int, ...], steps: int):
    json_results = {}
    for num_envs_pow in num_envs_pow2:
        num_envs = 2 ** num_envs_pow
        print(f"Benchmarking with {num_envs} envs...")
        total_time = env_loop(task, num_envs, steps, benchmark_interval=None)
        json_results[num_envs] = (steps / total_time) if total_time > 0 else -1
        if total_time > 0:
            print(f"Steps/s for {num_envs} envs: {json_results[num_envs]}")
        else:
            break
        time.sleep(5)
    return json_results

def benchmark_multiple_tasks(tasks_envs: dict[str, tuple[int, ...]], steps: int):
    all_results = {}
    for task in tasks_envs:
        print(f"Benchmarking task {task}...")
        all_results[task] = benchmark_task(task, tasks_envs[task], steps)
    return all_results

def main():
    tasks_envs = {
        "Isaac-Cartpole-Direct-v0": tuple(range(0, 15)),
        "Isaac-Ant-Direct-v0": tuple(range(0, 15)), 
        "Isaac-Repose-Cube-Shadow-Direct-v0": tuple(range(0, 15)),
    }
    steps = 10000
    results = benchmark_multiple_tasks(tasks_envs, steps)
    print("Final results:")
    print(results)

if __name__ == "__main__":
    main()
    
    
    