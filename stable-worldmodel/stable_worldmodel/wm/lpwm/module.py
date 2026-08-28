"""LpWM-specific projection modules and shared LeWM dynamics modules."""

import torch
import torch.nn.functional as F
from torch import nn

from stable_worldmodel.wm.lewm.module import Embedder, Predictor


class RepReLU(nn.Module):
    """Return exact ReLU features while differentiating through GELU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu = F.gelu(x)
        relu = F.relu(x)
        return gelu - gelu.detach() + relu.detach()


class MLP(nn.Module):
    """Two-layer projection head with an optional rectified output."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
        link_function_type='Identity',
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        links = {
            'Identity': nn.Identity,
            'ReLU': RepReLU,
        }
        if link_function_type not in links:
            raise ValueError(
                'link_function_type must be Identity or ReLU, '
                f'got {link_function_type!r}.'
            )

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim),
            links[link_function_type](),
        )

    def forward(self, x):
        return self.net(x)


__all__ = ['Embedder', 'MLP', 'Predictor', 'RepReLU']
