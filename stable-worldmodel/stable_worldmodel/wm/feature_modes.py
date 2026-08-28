"""Canonical global feature policy for the native JEPA baselines."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stable_worldmodel.wm.patch_aggregator import PatchFeatureAggregator


class FeatureMode(nn.Module):
    """Select the actual CLS token or one named patch-token readout."""

    GLOBAL_AGGREGATIONS = frozenset({'cls', 'mean', 'mlp', 'spatial_8x8'})

    def __init__(
        self,
        mode: str,
        aggregation: str,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
        num_patches: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.aggregation = str(aggregation)
        if self.mode != 'global':
            raise ValueError(
                'Native JEPA baselines require one global feature per frame; '
                f'got mode={self.mode!r}.'
            )
        if self.aggregation not in self.GLOBAL_AGGREGATIONS:
            valid = ', '.join(sorted(self.GLOBAL_AGGREGATIONS))
            raise ValueError(
                'global feature mode requires aggregation to be one of '
                f'{{{valid}}}, got {self.aggregation!r}.'
            )
        if self.aggregation == 'cls':
            self.aggregator = None
            return
        if input_dim is None or output_dim is None:
            raise ValueError(
                'global patch aggregation requires input_dim and output_dim.'
            )
        self.aggregator = PatchFeatureAggregator(
            mode=self.aggregation,
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            num_patches=num_patches,
            hidden_dim=hidden_dim,
        )

    @property
    def is_global(self) -> bool:
        return True

    def forward(self, output: Any) -> torch.Tensor:
        tokens = (
            output.last_hidden_state
            if hasattr(output, 'last_hidden_state')
            else output
        )
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            shape = getattr(tokens, 'shape', None)
            raise ValueError(
                'FeatureMode expects ViT tokens shaped (N, 1+P, D), '
                f'got {shape}.'
            )
        if tokens.size(1) < 2:
            raise ValueError(
                'FeatureMode expects one CLS token followed by at least one '
                f'patch token, got shape {tuple(tokens.shape)}.'
            )
        if self.aggregation == 'cls':
            return tokens[:, 0]
        return self.aggregator(tokens[:, 1:])


__all__ = ['FeatureMode']
