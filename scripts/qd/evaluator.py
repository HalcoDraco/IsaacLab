from batched_policy import BatchedPolicy
import torch
from typing import Callable, Optional

import jax
import jax.numpy as jnp


class JaxEvaluator:
    """Evaluates batched JAX (Flax) policies in an IsaacLab vectorized environment.

    Uses DLPack for zero-copy JAX <-> PyTorch GPU transfers each simulation step.

    Args:
        env: a vectorized IsaacLab gymnasium environment.
        num_envs: number of parallel environments (= number of policies).
        policy_network: a Flax nn.Module policy (e.g. qdax MLP).
        num_steps: max number of simulation steps per evaluation episode.
        descriptor_fn: optional callable(obs_accumulator, step_count, cumulative_rewards)
            -> descriptors tensor of shape (num_envs, num_descriptors).
            If None, defaults to the time-averaged mean of the first 2 observation dims.
    """

    def __init__(
        self,
        env,
        num_envs: int,
        policy_network,
        num_steps: int = 100,
        descriptor_fn: Optional[Callable] = None,
    ):
        self.env = env
        self.num_envs = num_envs
        self.policy_network = policy_network
        self.num_steps = num_steps
        self.descriptor_fn = descriptor_fn
        self._batched_apply = jax.jit(jax.vmap(policy_network.apply))

    def evaluate(self, genotypes) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a batch of JAX PyTree genotypes in the environment.

        Rolls out each policy for up to ``num_steps`` steps using a Flax
        network forward pass (JAX), with DLPack zero-copy transfers for
        observations and actions between PyTorch (IsaacLab) and JAX.

        Args:
            genotypes: JAX PyTree where each leaf has shape (num_policies, ...).

        Returns:
            fitnesses: torch tensor, shape (num_policies,) — cumulative reward.
            descriptors: torch tensor, shape (num_policies, num_descriptors).
        """
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        device = obs.device

        cumulative_rewards = torch.zeros(self.num_envs, device=device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        step_count = torch.zeros(self.num_envs, device=device)
        obs_accumulator = torch.zeros_like(obs)

        with torch.no_grad():
            for _ in range(self.num_steps):
                # JAX forward pass: obs (torch) -> JAX -> Flax policy -> actions (JAX) -> torch
                obs_jax = jnp.from_dlpack(obs.contiguous())
                actions_jax = self._batched_apply(genotypes, obs_jax)
                actions = torch.from_dlpack(actions_jax)

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
            # Default: time-averaged mean of the first 2 observation dims.
            mean_obs = obs_accumulator / step_count.unsqueeze(-1).clamp(min=1)
            descriptors = mean_obs[:, :2]

        return cumulative_rewards, descriptors