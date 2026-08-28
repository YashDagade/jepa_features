"""Paper-faithful module unique to the Sensorimotor World Model."""

from torch import nn


class InverseModel(nn.Module):
    """Predict an action from the ordered pair ``[z_t, z_{t+1}]``.

    The two 256-wide ReLU layers reproduce the official implementation at
    commit ``9d22bcc``.

    Source: https://github.com/petr-ivashkov/sensorimotor-world-model
    """

    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        if int(input_dim) <= 0 or int(input_dim) % 2:
            raise ValueError(
                'InverseModel input_dim must be a positive 2D width.'
            )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state_pair):
        return self.net(state_pair)


__all__ = ['InverseModel']
