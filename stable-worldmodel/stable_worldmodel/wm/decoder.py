"""Small diagnostic decoder for one-token/global visual embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalDecoder(nn.Module):
    """Decode one global embedding into an image through learned patch queries.

    This is a diagnostic head, not a generative model. With ``detach=true`` in
    the train config, reconstruction gradients train only this decoder, so the
    decoded clips show what is already linearly/attention-decodable from the
    latent without changing the encoder.
    """

    def __init__(
        self,
        global_dim: int = 384,
        img_size: int = 224,
        patch_size: int = 14,
        dim: int = 256,
        heads: int = 8,
        depth: int = 4,
    ):
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError('img_size must be divisible by patch_size')

        # 1. Derive the output patch grid and RGB width from image geometry.
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        patch_dim = patch_size * patch_size * 3

        # 2. Learn one image-space query for each output patch location.
        self.queries = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.normal_(self.queries, std=0.02)

        # 3. Project the global feature into the decoder's attention width.
        # Keep ``cls_proj`` as the state-dict key so existing compatible
        # decoder sidecars still load exactly after the public rename.
        self.cls_proj = nn.Sequential(
            nn.Linear(global_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # 4. Refine every query by cross-attending to the global feature.
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'cross_attn': nn.MultiheadAttention(
                            dim,
                            heads,
                            batch_first=True,
                        ),
                        'norm1': nn.LayerNorm(dim),
                        'mlp': nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                        'norm2': nn.LayerNorm(dim),
                    }
                )
                for _ in range(depth)
            ]
        )

        # 5. Decode each refined query into one RGB patch.
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode flattened global embeddings ``(B, D)`` into RGB images."""
        # 1. Expand learned spatial queries and project the global conditioning token.
        batch_size = x.size(0)
        queries = self.queries.expand(batch_size, -1, -1)
        keys_values = self.cls_proj(x).unsqueeze(1)

        # 2. Alternate global cross-attention and per-query MLP refinement.
        for layer in self.layers:
            attn_out = layer['cross_attn'](queries, keys_values, keys_values)[
                0
            ]
            queries = layer['norm1'](queries + attn_out)
            mlp_out = layer['mlp'](queries)
            queries = layer['norm2'](queries + mlp_out)

        # 3. Decode each query and restore the two-dimensional patch grid.
        patches = self.to_pixels(queries)
        patches = patches.reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            3,
        )
        # 4. Interleave grid and within-patch axes into one RGB image.
        image = patches.permute(0, 5, 1, 3, 2, 4)
        return image.reshape(
            batch_size,
            3,
            self.grid_size * self.patch_size,
            self.grid_size * self.patch_size,
        )


__all__ = ['GlobalDecoder']
