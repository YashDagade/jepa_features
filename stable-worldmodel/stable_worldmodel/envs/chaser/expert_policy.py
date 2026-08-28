"""Shortest-direction expert for Chaser dataset collection."""

from __future__ import annotations

from typing import Any

import numpy as np

from stable_worldmodel.policy import BasePolicy


class ChaserExpertPolicy(BasePolicy):
    """Move the red agent toward the visible blue target."""

    def __init__(self, action_noise: float = 0.0, seed: int | None = None):
        super().__init__()
        if action_noise < 0:
            raise ValueError('action_noise must be non-negative.')
        self.type = 'chaser_expert'
        self.action_noise = float(action_noise)
        self.set_seed(seed)

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_action(
        self, info_dict: dict[str, Any], **_kwargs: Any
    ) -> np.ndarray:
        """Return normalized red-to-blue directions for scalar or pooled envs."""
        vectorized = hasattr(self.env, 'envs') or hasattr(
            getattr(self.env, 'unwrapped', None), 'envs'
        )
        red = np.asarray(info_dict['pos_red'], dtype=np.float32)
        blue = np.asarray(info_dict['pos_blue'], dtype=np.float32)
        if vectorized:
            red = red.reshape(red.shape[0], -1)
            blue = blue.reshape(blue.shape[0], -1)
        else:
            red = red.reshape(1, -1)
            blue = blue.reshape(1, -1)

        direction = blue - red
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        actions = direction / np.maximum(norm, 1e-8)
        if self.action_noise:
            actions += self.rng.normal(
                0.0, self.action_noise, size=actions.shape
            ).astype(np.float32)
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
        return actions if vectorized else actions[0]
