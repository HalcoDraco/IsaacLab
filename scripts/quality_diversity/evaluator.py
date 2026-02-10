"""GPU-batched evaluation of multiple policies in an IsaacLab environment.

The evaluator rolls out ``num_envs`` copies of the environment simultaneously.
Each environment instance executes a *different* policy (identified by its flat
parameter vector).  The rollout runs entirely on the GPU; only the final
(objective, measures) arrays are transferred to CPU/numpy so pyribs can
consume them.

Design: the environment auto-resets terminated sub-environments, but we
accumulate reward only for the **first** episode of each env.  This way
``num_envs`` policies are each scored for exactly one episode.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from policy import MLPPolicy, make_batched_forward, count_params
from measures import MeasureFunction


# Type alias for the transition callback used by PGA-MAP-Elites.
TransitionCallback = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    None,
]


class Evaluator:
    """Evaluate a batch of flat parameter vectors in an IsaacLab env.

    Parameters
    ----------
    env : gymnasium.Env
        A vectorised IsaacLab environment (``gym.make(...)``).
    obs_dim, action_dim : int
        Observation / action dimensions.
    hidden : tuple[int, ...]
        Hidden-layer sizes for the MLP policy.
    measure_fn : MeasureFunction
        Computes behavioural descriptors from the rollout.
    max_steps : int
        Hard cap on simulation steps per evaluation batch.
    """

    def __init__(
        self,
        env,
        obs_dim: int,
        action_dim: int,
        hidden: tuple[int, ...],
        measure_fn: MeasureFunction,
        max_steps: int = 1000,
    ):
        self.env = env
        self.device = torch.device(env.unwrapped.device)
        self.num_envs = env.unwrapped.num_envs
        self.max_steps = max_steps
        self.measure_fn = measure_fn

        # Build a template policy (weights irrelevant — only structure matters).
        template = MLPPolicy(obs_dim, action_dim, hidden).to(self.device)
        self.param_dim = count_params(template)
        self._batched_fwd, _, _ = make_batched_forward(template, self.device)

    # ---- public API -------------------------------------------------------

    def evaluate(
        self,
        flat_params_np: np.ndarray,
        transition_cb: TransitionCallback | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of solutions.

        Parameters
        ----------
        flat_params_np : shape ``(batch_size, param_dim)``
            Flattened policy parameters coming from pyribs.
        transition_cb : optional
            If provided, called at every step with
            ``(obs, actions, rewards, next_obs, dones)`` for envs still in
            their first episode.  Used by PGA-MAP-Elites to fill the replay
            buffer without duplicating the rollout.

        Returns
        -------
        objectives : shape ``(batch_size,)``
        measures   : shape ``(batch_size, measure_dim)``
        """
        batch_size = flat_params_np.shape[0]
        assert batch_size <= self.num_envs

        # Transfer parameters to GPU once per generation.
        stacked = torch.as_tensor(                          # (batch, param_dim)
            flat_params_np, dtype=torch.float32, device=self.device,
        )
        if batch_size < self.num_envs:
            pad = torch.zeros(                              # (pad, param_dim)
                self.num_envs - batch_size, self.param_dim,
                dtype=torch.float32, device=self.device,
            )
            stacked = torch.cat([stacked, pad], dim=0)      # (N, param_dim)

        obj, meas = self._rollout(stacked, transition_cb)   # (N,), (N, measure_dim)
        return obj[:batch_size].cpu().numpy(), meas[:batch_size].cpu().numpy()

    # ---- internal ---------------------------------------------------------

    @torch.no_grad()
    def _rollout(
        self,
        stacked_params: torch.Tensor,
        transition_cb: TransitionCallback | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-episode rollout, entirely on GPU."""
        N = self.num_envs
        obs = self.env.reset()[0]["policy"]            # (N, obs_dim)

        cum_reward = torch.zeros(N, device=self.device)                # (N,)
        done_mask  = torch.zeros(N, dtype=torch.bool, device=self.device)  # (N,)
        self.measure_fn.reset(N, self.device)

        for _ in range(self.max_steps):
            actions = self._batched_fwd(stacked_params, obs)  # (N, action_dim)
            obs_dict, rewards, terminated, truncated, info = self.env.step(actions)
            next_obs = obs_dict["policy"]                     # (N, obs_dim)
            dones = terminated | truncated                     # (N,)

            # Collect transitions for active (first-episode) envs.
            if transition_cb is not None:
                active = ~done_mask
                if active.any():
                    idx = active.nonzero(as_tuple=True)[0]
                    transition_cb(
                        obs[idx], actions[idx], rewards[idx],
                        next_obs[idx], dones[idx].float(),
                    )

            cum_reward += rewards * (~done_mask).float()
            self.measure_fn.update(next_obs, actions, rewards,
                                   terminated, truncated, info)
            done_mask |= dones.bool()
            obs = next_obs

            if done_mask.all():
                break

        return cum_reward, self.measure_fn.compute()
