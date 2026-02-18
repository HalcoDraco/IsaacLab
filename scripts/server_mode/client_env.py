import torch
from socket_env.socket_env_client import SocketEnvClient

def main():
    env = SocketEnvClient()
    env.make(task="Isaac-Cartpole-Direct-v0", num_envs=64)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs = env.reset()

    cumulative_rewards = torch.zeros(env.num_envs, device=env.device)
    dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    step_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    # simulate environment
    for step in range(1000):
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.device) - 1
            # apply actions
            obs, rewards, terminated, truncated = env.step(actions)

            alive = ~dones
            cumulative_rewards += rewards * alive.float()
            step_count += alive.int()

            dones |= (terminated | truncated)

            # Print dones for debugging
            print(f"[client_env] Step {step+1}: dones = {dones.cpu().numpy()}")

            if dones.all():
                break

    # close the simulator
    env.close()

if __name__ == "__main__":
    main()
    