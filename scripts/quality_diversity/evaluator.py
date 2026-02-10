"""GPU-batched evaluation of multiple policies in an IsaacLab environment.

The evaluator rolls out ``num_envs`` copies of the environment simultaneously.
Each environment instance executes a *different* policy (identified by its flat
parameter vector).  The rollout runs entirely on the GPU; only the final
(objective, measures) arrays are transferred to CPU/numpy so pyribs can
consume them.

Design: the environment auto-resets terminated sub-environments, but we
accumulate reward only for the **first** episode of each env.  This way
``num_envs`` policies are each scored for exactly one episode.

Transitions for the PGA critic are accumulated on-GPU during the rollout and
returned in bulk — no per-step GPU→CPU synchronisation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from measures import MeasureFunction
from policy import MLPPolicy, count_params, make_batched_forward


# Type alias for the bulk transition callback used by PGA-MAP-Elites.
# Called *once* after rollout with all first-episode transitions.
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
        self.obs_dim = obs_dim
        self.action_dim = action_dim

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
            If provided, called **once** after the rollout with the full
            buffer of first-episode transitions collected on-GPU.

        Returns
        -------
        objectives : shape ``(batch_size,)``
        measures   : shape ``(batch_size, measure_dim)``
        """
        batch_size = flat_params_np.shape[0]
        assert batch_size <= self.num_envs

        # Transfer parameters to GPU once per generation.
        stacked = torch.as_tensor(
            flat_params_np, dtype=torch.float32, device=self.device,
        )
        if batch_size < self.num_envs:
            pad = torch.zeros(
                self.num_envs - batch_size, self.param_dim,
                dtype=torch.float32, device=self.device,
            )
            stacked = torch.cat([stacked, pad], dim=0)

        collect = transition_cb is not None
        obj, meas, transitions = self._rollout(stacked, collect)

        # Bulk-transfer transitions to the PGA emitter (single GPU→CPU copy).
        if transition_cb is not None and transitions is not None:
            transition_cb(*transitions)

        return obj[:batch_size].cpu().numpy(), meas[:batch_size].cpu().numpy()

    # ---- internal ---------------------------------------------------------

    @torch.no_grad()
    def _rollout(
        self,
        stacked_params: torch.Tensor,
        collect_transitions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
        """One-episode rollout, entirely on GPU.

        When *collect_transitions* is True, first-episode transitions are
        accumulated in pre-allocated GPU buffers and returned in bulk —
        avoiding the per-step GPU→CPU sync that was the main bottleneck.
        """
        N = self.num_envs
        obs = self.env.reset()[0]["policy"]  # (N, obs_dim)

        cum_reward = torch.zeros(N, device=self.device)
        done_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.measure_fn.reset(N, self.device)

        # Pre-allocate GPU transition buffers (only if needed).
        if collect_transitions:
            buf_obs = torch.empty(self.max_steps, N, self.obs_dim, device=self.device)
            buf_act = torch.empty(self.max_steps, N, self.action_dim, device=self.device)
            buf_rew = torch.empty(self.max_steps, N, device=self.device)
            buf_nobs = torch.empty(self.max_steps, N, self.obs_dim, device=self.device)
            buf_done = torch.empty(self.max_steps, N, device=self.device)
            buf_mask = torch.empty(self.max_steps, N, dtype=torch.bool, device=self.device)

        step = 0
        for step in range(self.max_steps):
            actions = self._batched_fwd(stacked_params, obs)
            obs_dict, rewards, terminated, truncated, info = self.env.step(actions)
            next_obs = obs_dict["policy"]
            dones = terminated | truncated

            # Only accumulate first-episode data.
            active = ~done_mask

            if collect_transitions:
                buf_obs[step] = obs
                buf_act[step] = actions
                buf_rew[step] = rewards
                buf_nobs[step] = next_obs
                buf_done[step] = dones.float()
                buf_mask[step] = active

            cum_reward += rewards * active.float()
            self.measure_fn.update(next_obs, actions, rewards,
                                   terminated, truncated, info, active)
            done_mask |= dones.bool()
            obs = next_obs

            if done_mask.all():
                break

        T = step + 1  # actual number of steps taken

        # Flatten and filter transitions (GPU-only, single bulk operation).
        transitions = None
        if collect_transitions:
            mask_flat = buf_mask[:T].reshape(-1)
            if mask_flat.any():
                idx = mask_flat.nonzero(as_tuple=True)[0]
                transitions = (
                    buf_obs[:T].reshape(-1, self.obs_dim)[idx],
                    buf_act[:T].reshape(-1, self.action_dim)[idx],
                    buf_rew[:T].reshape(-1)[idx],
                    buf_nobs[:T].reshape(-1, self.obs_dim)[idx],
                    buf_done[:T].reshape(-1)[idx],
                )

        return cum_reward, self.measure_fn.compute(), transitions
