import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    obs_dict, _ = env.reset()
    sim_dones = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            obs_dict, reward, terminated, truncated, extras = env.step(actions)
            dones = terminated | truncated
            sim_dones |= dones
            # print("Obs:", obs_dict, "Reward:", reward, "Terminated:", terminated, "Truncated:", truncated, "Extras:", extras)
            print(f"Rewards: {reward}, Dones: {sim_dones}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()



    policy = Policy(obs_dim=10, hidden=64, action_dim=4)
    print("Policy parameters:")
    named_params = policy.named_parameters()
    for name, param in named_params:
        print(f"{name}: {param.shape}")

    named_buffers = policy.named_buffers()
    print("\nPolicy buffers:")
    for name, buffer in named_buffers:
        print(f"{name}: {buffer.shape}")

    list_named_params_shape = [(name, param.shape) for name, param in policy.named_parameters()]
    print("\nList of named parameters and their shapes:")
    for name, shape in list_named_params_shape:
        print(f"{name}: {shape}")

    # Example of flattening parameters into a 1D vector
    flat_params = torch.cat([param.flatten() for param in policy.parameters()])
    print(f"\nFlattened parameters shape: {flat_params.shape}")

    # Example of unflattening back to original shapes
    param_shapes = [param.shape for param in policy.parameters()]
    unflattened_named_params = {}
    offset = 0
    for name, shape in list_named_params_shape:
        numel = torch.prod(torch.tensor(shape)).item()
        unflattened_named_params[name] = flat_params[offset : offset + numel].reshape(shape)
        offset += numel

    print("\nUnflattened parameters:")
    for name, param in unflattened_named_params.items():
        print(f"{name}: {param.shape}")

    # Verify that unflattening gives the same parameters back
    for name, param in policy.named_parameters():
        assert torch.allclose(param, unflattened_named_params[name]), f"Mismatch in parameter {name}"
    print("\nUnflattening verified to match original parameters.")


def copied(self):
    param_names: list[str] = []
    param_shapes: list[tuple[int, ...]] = []
    for name, p in model.named_parameters():
        param_names.append(name)
        param_shapes.append(p.shape)

    def _unflatten(flat_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unflatten a 1-D vector into a {name: tensor} dict."""
        params: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape in zip(param_names, param_shapes):
            n = 1
            for s in shape:
                n *= s
            params[name] = flat_vec[offset : offset + n].reshape(shape)
            offset += n
        return params

    # Single-policy forward: (flat_params_1d, obs_single) → action_single
    def _single_forward(flat_params: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        params = _unflatten(flat_params)
        return torch.func.functional_call(model, params, obs)

    # Vectorize over the first dimension of both flat_params and obs.
    batched_forward = torch.vmap(_single_forward, in_dims=(0, 0))

    return batched_forward, param_shapes, param_names