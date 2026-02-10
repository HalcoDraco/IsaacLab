"""Simple MLP policy with vectorized (vmap) evaluation support.

The policy is a small feed-forward network whose parameters can be
flattened to / unflattened from a 1-D vector. This lets pyribs treat
each solution as a flat numpy array while we evaluate many policies
simultaneously on the GPU via ``torch.vmap`` + ``torch.func.functional_call``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class MLPPolicy(nn.Module):
    """Deterministic MLP policy: obs → action (tanh-squashed)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # (*, obs_dim) → (*, action_dim)


# ---------------------------------------------------------------------------
# Parameter utilities
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    """Return the total number of (flat) parameters."""
    return sum(p.numel() for p in model.parameters())


def flat_params(model: nn.Module) -> torch.Tensor:
    """Return all parameters as a single 1-D tensor (detached, on same device)."""
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])  # (param_dim,)


def load_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Write a flat parameter vector back into the model's buffers."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].reshape(p.shape))
        offset += n


# ---------------------------------------------------------------------------
# Vectorized (batched) forward pass
# ---------------------------------------------------------------------------

def make_batched_forward(model: nn.Module, device: torch.device):
    """Build a vmapped function: (stacked_params, obs_batch) → actions.

    Parameters
    ----------
    model : MLPPolicy
        A *single* policy instance used as the architecture template.
        Its current parameter values are irrelevant.
    device : torch.device
        Device where both params and observations live.

    Returns
    -------
    batched_forward : callable
        ``batched_forward(stacked_params, obs)`` where
        *stacked_params* has shape ``(num_policies, param_dim)`` and
        *obs* has shape ``(num_policies, obs_dim)``.
        Returns actions of shape ``(num_policies, action_dim)``.
    param_shapes : list[tuple[int, ...]]
        Shapes of each parameter tensor (needed for unflattening).
    param_names : list[str]
        Corresponding parameter names.
    """
    # Snapshot the structure (shapes + names) of the model's parameters.
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
