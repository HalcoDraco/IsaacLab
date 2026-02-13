from batched_policy import BatchedPolicy
import torch
from typing import Callable, Optional


class Evaluator:
    """Evaluates a batch of policies in an IsaacLab vectorized environment.

    Each policy is assigned to one environment instance. All policies are
    evaluated in parallel using the BatchedPolicy's vmapped forward pass.

    Args:
        env: a vectorized IsaacLab gymnasium environment.
        num_envs: number of parallel environments (= number of policies).
        model: a torch.nn.Module policy (used as architecture template).
        num_steps: max number of simulation steps per evaluation episode.
        descriptor_fn: optional callable(obs_accumulator, step_count, cumulative_rewards)
            -> descriptors tensor of shape (num_envs, num_descriptors).
            If None, defaults to the time-averaged mean of the first 2 observation dims.
    """

    def __init__(
        self,
        env,
        num_envs: int,
        model: torch.nn.Module,
        num_steps: int = 100,
        descriptor_fn: Optional[Callable] = None,
    ):
        self.env = env
        self.num_envs = num_envs
        self.batched_policy = BatchedPolicy(model, num_envs)
        self.num_steps = num_steps
        self.descriptor_fn = descriptor_fn

    def evaluate(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a batch of flat parameter vectors in the environment.

        Rolls out each policy for up to ``num_steps`` steps (or until the
        episode terminates), accumulating rewards as fitness and computing
        behavioral descriptors.

        Args:
            parameters: tensor of shape (num_policies, num_params).

        Returns:
            fitnesses: shape (num_policies,) — cumulative reward per episode.
            descriptors: shape (num_policies, num_descriptors).
        """
        self.batched_policy.set_population_parameters(parameters)

        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        device = obs.device

        cumulative_rewards = torch.zeros(self.num_envs, device=device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        step_count = torch.zeros(self.num_envs, device=device)
        obs_accumulator = torch.zeros_like(obs)

        with torch.inference_mode():
            for _ in range(self.num_steps):
                actions = self.batched_policy.batched_forward(obs)
                obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
                obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

                alive = ~dones
                cumulative_rewards += rewards * alive.float()
                obs_accumulator += obs * alive.float().unsqueeze(-1)
                step_count += alive.float()

                dones |= (terminated | truncated)

                if dones.all():
                    break

        # Compute descriptors
        if self.descriptor_fn is not None:
            descriptors = self.descriptor_fn(
                obs_accumulator, step_count, cumulative_rewards
            )
        else:
            # Default: time-averaged mean of the first 2 observation dimensions.
            # Replace this with a task-specific descriptor for meaningful QD results.
            mean_obs = obs_accumulator / step_count.unsqueeze(-1).clamp(min=1)
            descriptors = mean_obs[:, :2]

        return cumulative_rewards, descriptors