"""Strict global readouts over ViT patch tokens."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PatchFeatureAggregator(nn.Module):
    """Compress ``(N, P, D_in)`` patch tokens into one frame vector."""

    MODES = frozenset({'mean', 'mlp', 'spatial_8x8'})

    def __init__(
        self,
        mode: str,
        input_dim: int,
        output_dim: int,
        *,
        num_patches: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.mode not in self.MODES:
            valid = ', '.join(sorted(self.MODES))
            raise ValueError(
                f'PatchFeatureAggregator mode must be one of {{{valid}}}, '
                f'got {self.mode!r}.'
            )
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError('input_dim and output_dim must both be positive.')
        self.num_patches = None if num_patches is None else int(num_patches)
        self.hidden_dim = None if hidden_dim is None else int(hidden_dim)

        if self.mode == 'mean':
            self.proj = (
                nn.Identity()
                if self.input_dim == self.output_dim
                else nn.Linear(self.input_dim, self.output_dim)
            )
        elif self.mode == 'spatial_8x8':
            pooled_dim = 8 * 8 * self.input_dim
            self.proj = nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Linear(pooled_dim, self.output_dim),
            )
        else:
            if self.num_patches is None or self.num_patches <= 0:
                raise ValueError(
                    'num_patches must be a positive integer for mlp '
                    'aggregation.'
                )
            if self.hidden_dim is None:
                self.hidden_dim = 4 * self.output_dim
            if self.hidden_dim <= 0:
                raise ValueError('hidden_dim must be positive when provided.')
            self.proj = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(
                    self.num_patches * self.input_dim,
                    self.hidden_dim,
                ),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3:
            raise ValueError(
                'PatchFeatureAggregator expects (N, P, D) patch tokens, '
                f'got {tuple(patches.shape)}.'
            )
        if patches.size(-1) != self.input_dim:
            raise ValueError(
                'PatchFeatureAggregator feature width mismatch: '
                f'expected {self.input_dim}, got {patches.size(-1)}.'
            )
        if self.mode == 'mean':
            return self.proj(patches.mean(dim=1))
        if self.mode == 'mlp':
            if patches.size(1) != self.num_patches:
                raise ValueError(
                    'PatchFeatureAggregator patch count mismatch: '
                    f'expected {self.num_patches}, got {patches.size(1)}.'
                )
            return self.proj(patches)

        grid = math.isqrt(int(patches.size(1)))
        if grid * grid != patches.size(1):
            raise ValueError(
                f'Expected square patch grid, got {patches.size(1)} patches.'
            )
        if grid < 8:
            raise ValueError(
                'spatial_8x8 aggregation requires at least an 8x8 patch '
                f'grid, got {grid}x{grid}.'
            )
        patch_map = patches.reshape(
            patches.size(0), grid, grid, patches.size(-1)
        ).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(patch_map, output_size=(8, 8))
        return self.proj(pooled.flatten(1))


__all__ = ['PatchFeatureAggregator']
