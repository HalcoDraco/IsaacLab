from batched_policy import BatchedPolicy
import torch

class Evaluator:

    def __init__(self, env, num_envs, policy, num_episodes=100):
        self.env = env
        self.num_envs = num_envs
        self.policy = policy
        self.num_episodes = num_episodes

    def evaluate(self, parameters: torch.Tensor):
        pass