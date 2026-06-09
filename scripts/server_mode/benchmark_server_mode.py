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

def save_results(all_results: dict, output_path: str):
    # Best-effort: a write hiccup must not abort the benchmark run.
    try:
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(f"{all_results}\n")
    except OSError as exc:
        print(f"Warning: failed to persist partial results to {output_path}: {exc}")

def benchmark_task(
    task: str,
    num_envs_pow2: tuple[int, ...],
    steps: int,
    num_runs: int = 3,
    all_results: dict | None = None,
    output_path: str | None = None,
):
    json_results = {}
    # Expose this task's (initially empty) results through the shared dict so
    # partial results are persisted after every iteration below.
    if all_results is not None:
        all_results[task] = json_results
    for num_envs_pow in num_envs_pow2:
        num_envs = 2 ** num_envs_pow
        print(f"Benchmarking with {num_envs} envs...")
        run_times = []
        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs} for {num_envs} envs...")
            total_time = env_loop(task, num_envs, steps, benchmark_interval=None)
            run_times.append(total_time)
            if total_time > 0:
                print(f"Steps/s for run {run+1}: {steps / total_time}")
            else:
                print(f"Run {run+1} failed.")
                break
            time.sleep(8)
        if all(t > 0 for t in run_times):
            avg_time = sum(run_times) / num_runs
            json_results[num_envs] = steps / avg_time
            print(f"Average Steps/s for {num_envs} envs: {json_results[num_envs]}")
            failed = False
        else:
            json_results[num_envs] = -1
            print(f"Benchmarking failed for {num_envs} envs. Stopping further benchmarks.")
            failed = True

        # Persist partial results after each iteration so they survive a crash.
        if all_results is not None and output_path is not None:
            save_results(all_results, output_path)

        if failed:
            break

    return json_results

def benchmark_multiple_tasks(tasks_envs: dict[str, tuple[int, ...]], steps: int, output_path: str | None = None):
    all_results = {}
    for task in tasks_envs:
        print(f"Benchmarking task {task}...")
        all_results[task] = benchmark_task(
            task, tasks_envs[task], steps, all_results=all_results, output_path=output_path
        )
    return all_results

def main():
    tasks_envs = {
        "Isaac-Cartpole-Direct-v0": tuple(range(0, 15)),
        "Isaac-Ant-Direct-v0": tuple(range(0, 15)), 
        "Isaac-Repose-Cube-Shadow-Direct-v0": tuple(range(0, 15)),
    }
    steps = 10000
    output_path = "benchmark_server_mode_results.txt"
    results = benchmark_multiple_tasks(tasks_envs, steps, output_path=output_path)
    print("Final results:")
    print(results)
    save_results(results, output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
    
    
    