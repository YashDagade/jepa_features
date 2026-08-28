import torch
import torch.nn.functional as F
from einops import einsum


class InfoReg(torch.nn.Module):
    """Match projected features to a rectified generalized-normal target.

    LpWM uses ``p=1`` and ``mu=0`` so the target is a rectified Laplace
    distribution.  The input contract is ``(group, sample, feature)``; native
    global-token training uses one sample cloud per timestep.
    """

    def __init__(
        self,
        *,
        mu: float = 0.0,
        p: float = 1.0,
        link_function_type: str = 'ReLU',
        loss_func_type: str = 'sliced_wasserstein',
        num_slices: int = 1024,
        target_dist_rms_norm: bool = False,
    ):
        super().__init__()
        if p <= 0:
            raise ValueError(f'InfoReg p must be positive, got {p}.')
        if num_slices <= 0:
            raise ValueError('InfoReg num_slices must be positive.')
        if link_function_type not in {'Identity', 'ReLU'}:
            raise ValueError('link_function_type must be Identity or ReLU.')
        if loss_func_type != 'sliced_wasserstein':
            raise ValueError('Only sliced_wasserstein InfoReg is supported.')

        self.mu = float(mu)
        self.p = float(p)
        self.link_function_type = link_function_type
        self.num_slices = int(num_slices)
        self.target_dist_rms_norm = bool(target_dist_rms_norm)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """Return the sliced-Wasserstein distance for independent clouds."""
        # 1. Each group needs at least two samples to define an empirical cloud.
        if samples.ndim != 3 or samples.size(1) < 2:
            raise ValueError(
                'InfoReg expects (groups,samples,features) with at least '
                f'two samples, got {tuple(samples.shape)}.'
            )
        source = samples.float()

        # 2. Draw an equally shaped generalized-normal target using its Gamma
        # magnitude representation, then apply LpWM's configured output link.
        concentration = source.new_tensor(1.0 / self.p)
        magnitude = torch.distributions.Gamma(
            concentration,
            source.new_tensor(1.0),
        ).sample(source.shape)
        magnitude = magnitude.pow(1.0 / self.p)
        sign = torch.empty_like(source).bernoulli_(0.5).mul_(2).sub_(1)
        target = self.mu + sign * magnitude
        if self.link_function_type == 'ReLU':
            target = F.relu(target)
        if self.target_dist_rms_norm:
            target = target / target.square().mean(
                dim=-1, keepdim=True
            ).sqrt().clamp_min(1e-8)

        # 3. Compare sorted source and target projections along shared random
        # directions, which forms the empirical one-dimensional couplings.
        directions = F.normalize(
            torch.randn(
                source.size(-1),
                self.num_slices,
                device=source.device,
                dtype=source.dtype,
            ),
            dim=0,
        )
        source_proj = (source @ directions).sort(dim=1).values
        target_proj = (target @ directions).sort(dim=1).values
        return F.mse_loss(source_proj, target_proj)


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Warning: This version only support single-gpu.
    Reference: https://arxiv.org/abs/2511.08544
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device='cuda')
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(
            -3
        ).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


class VCReg(torch.nn.Module):
    """Variance-Covariance Regularizer"""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def _std_loss(self, z):
        z = z.transpose(0, 1)  # (T, B, D)
        std = (z.var(dim=1) + self.eps).sqrt()  # (T, D)
        std_loss = torch.mean(F.relu(1 - std), dim=-1)  # (T,)
        return std_loss

    def _cov_loss(self, z):
        B, T, D = z.shape
        z = z.transpose(0, 1)  # (T, B, D)
        cov = einsum(z, z, 't b i, t b j -> t i j') / (B - 1)  # (T, D, D)
        diag = einsum(cov, 't i i -> t i').pow(2).sum(dim=-1)  # (T,)
        cov_loss = (cov.pow(2).sum(dim=[-1, -2]) - diag).div(D**2 - D)  # (T,)
        return cov_loss

    def forward(self, z):
        """
        z: (..., D)
        """

        if z.dim() == 2:
            D = z.size(-1)
            z = z.view(-1, D)

        z = z - z.mean(
            dim=0, keepdim=True
        )  # mean for each dim across batch samples

        return {
            'std_loss': self._std_loss(z).mean(),
            'std_t_loss': self._std_loss(z.transpose(0, 1)).mean(),
            'cov_loss': self._cov_loss(z).mean(),
            'cov_t_loss': self._cov_loss(z.transpose(0, 1)).mean(),
        }


class PLDMLoss(torch.nn.Module):
    """PLDM variance/covariance and temporal-alignment regularizers.

    reference: https://arxiv.org/abs/2502.14819
    """

    def __init__(self):
        super().__init__()
        self.vc_reg = VCReg()

    def forward(self, z):
        """
        z: (B, T, D)
        """

        output = {'temp_align_loss': F.mse_loss(z[:, :-1], z[:, 1:])}
        output.update(self.vc_reg(z))

        return output


__all__ = [
    'InfoReg',
    'PLDMLoss',
    'SIGReg',
    'VCReg',
]
