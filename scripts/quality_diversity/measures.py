"""Modular behavioral-descriptor (measure) definitions.

Each ``MeasureFunction`` accumulates per-step data during a rollout and
reduces it to a fixed-size descriptor vector at the end of the episode.
Multiple measures can be composed via ``CompositeMeasure``.

All operations stay on the GPU — no CPU transfers happen here.

The ``update`` method receives an *active* mask so that only first-episode
data is accumulated (environments auto-reset after termination).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class MeasureFunction(ABC):
    """Interface for one behavioral descriptor component."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the measure vector this function produces."""

    @abstractmethod
    def reset(self, num_envs: int, device: torch.device) -> None:
        """Reset internal accumulators (called once before a rollout batch)."""

    @abstractmethod
    def update(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
               terminated: torch.Tensor, truncated: torch.Tensor,
               info: dict, active: torch.Tensor) -> None:
        """Accumulate data from one environment step.

        Parameters
        ----------
        active : (N,) bool tensor
            True for envs still in their first episode.
        """

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Return the final measure vector, shape ``(num_envs, self.dim)``."""


# ---------------------------------------------------------------------------
# Concrete measures
# ---------------------------------------------------------------------------

class FinalXYPosition(MeasureFunction):
    """Measure = final (x, y) position of the root body.

    Requires ``info`` to contain ``"root_pos"`` of shape ``(num_envs, 3)``,
    or falls back to accumulating last non-reset position.
    """

    @property
    def dim(self) -> int:
        return 2

    def reset(self, num_envs: int, device: torch.device) -> None:
        self._pos = torch.zeros(num_envs, 3, device=device)  # (N, 3)

    def update(self, obs, actions, rewards, terminated, truncated, info, active):
        if "root_pos" in info:
            # Only update positions for envs still in their first episode.
            mask = active.unsqueeze(-1)  # (N, 1)
            self._pos = torch.where(mask, info["root_pos"], self._pos)

    def compute(self) -> torch.Tensor:
        return self._pos[:, :2]  # (N, 2)


class MeanXYVelocity(MeasureFunction):
    """Measure = time-averaged (vx, vy) of the root body.

    Requires ``info`` to contain ``"root_lin_vel"`` of shape ``(num_envs, 3)``.
    """

    @property
    def dim(self) -> int:
        return 2

    def reset(self, num_envs: int, device: torch.device) -> None:
        self._vel_sum = torch.zeros(num_envs, 3, device=device)  # (N, 3)
        self._count = torch.zeros(num_envs, 1, device=device)    # (N, 1)

    def update(self, obs, actions, rewards, terminated, truncated, info, active):
        if "root_lin_vel" in info:
            mask = active.unsqueeze(-1)  # (N, 1)
            self._vel_sum += info["root_lin_vel"] * mask
            self._count += active.unsqueeze(-1).float()

    def compute(self) -> torch.Tensor:
        mean_vel = self._vel_sum / self._count.clamp(min=1)  # (N, 3)
        return mean_vel[:, :2]  # (N, 2)


class ObservationSlice(MeasureFunction):
    """Measure = time-averaged slice of the observation vector.

    A generic fallback when env-specific info is not available.
    E.g. for Cartpole: indices [0, 2] → (pole_angle, cart_pos).
    """

    def __init__(self, indices: list[int]):
        self._indices = indices

    @property
    def dim(self) -> int:
        return len(self._indices)

    def reset(self, num_envs: int, device: torch.device) -> None:
        self._sum = torch.zeros(num_envs, self.dim, device=device)  # (N, dim)
        self._count = torch.zeros(num_envs, 1, device=device)       # (N, 1)

    def update(self, obs, actions, rewards, terminated, truncated, info, active):
        mask = active.unsqueeze(-1)  # (N, 1)
        self._sum += obs[:, self._indices] * mask
        self._count += active.unsqueeze(-1).float()

    def compute(self) -> torch.Tensor:
        return self._sum / self._count.clamp(min=1)  # (N, dim)


class MeanActionMagnitude(MeasureFunction):
    """Measure = mean |action| per action dimension, reduced to 1-D."""

    @property
    def dim(self) -> int:
        return 1

    def reset(self, num_envs: int, device: torch.device) -> None:
        self._sum = torch.zeros(num_envs, device=device)    # (N,)
        self._count = torch.zeros(num_envs, device=device)  # (N,)

    def update(self, obs, actions, rewards, terminated, truncated, info, active):
        self._sum += actions.abs().mean(dim=-1) * active.float()
        self._count += active.float()

    def compute(self) -> torch.Tensor:
        return (self._sum / self._count.clamp(min=1)).unsqueeze(-1)  # (N, 1)


class CartpoleMeasure(MeasureFunction):
    """Measure for IsaacLab Cartpole: (mean cart pos, mean |cart vel|).

    IsaacLab Cartpole observation layout:
        [0] pole angle       (rad)
        [1] pole angular vel (rad/s)
        [2] cart position    (m)
        [3] cart velocity    (m/s)

    The two behavioural descriptors are:
        - Time-averaged cart position   (obs index 2).
        - Time-averaged |cart velocity| (|obs index 3|).
    """

    @property
    def dim(self) -> int:
        return 2

    def reset(self, num_envs: int, device: torch.device) -> None:
        self._pos_sum = torch.zeros(num_envs, device=device)  # (N,)
        self._vel_sum = torch.zeros(num_envs, device=device)  # (N,)
        self._count = torch.zeros(num_envs, device=device)    # (N,)

    def update(self, obs, actions, rewards, terminated, truncated, info, active):
        # obs: (N, 4) — only accumulate for first-episode envs.
        a = active.float()
        self._pos_sum += obs[:, 2] * a
        self._vel_sum += obs[:, 3].abs() * a
        self._count += a

    def compute(self) -> torch.Tensor:
        c = self._count.clamp(min=1)  # (N,)
        mean_pos = self._pos_sum / c  # (N,)
        mean_vel = self._vel_sum / c  # (N,)
        return torch.stack([mean_pos, mean_vel], dim=-1)  # (N, 2)


# ---------------------------------------------------------------------------
# Composite: concatenate several measures
# ---------------------------------------------------------------------------

class CompositeMeasure(MeasureFunction):
    """Concatenate multiple measures into one vector."""

    def __init__(self, measures: list[MeasureFunction]):
        self._measures = measures

    @property
    def dim(self) -> int:
        return sum(m.dim for m in self._measures)

    def reset(self, num_envs: int, device: torch.device) -> None:
        for m in self._measures:
            m.reset(num_envs, device)

    def update(self, obs, actions, rewards, terminated, truncated, info, active):
        for m in self._measures:
            m.update(obs, actions, rewards, terminated, truncated, info, active)

    def compute(self) -> torch.Tensor:
        parts = [m.compute() for m in self._measures]
        return torch.cat(parts, dim=-1)
