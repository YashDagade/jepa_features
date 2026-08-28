"""Feature diagnostics for spotting latent collapse during training."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F


def flatten_feature_batch(feats: torch.Tensor) -> torch.Tensor:
    """Flatten all non-feature axes into one batch axis.

    The world models use both CLS features ``(B, T, D)`` and patch features
    ``(B, T, P, D)``. Diagnostics care only about the feature cloud, so every
    axis before the final feature dimension becomes one large sample axis.
    """
    if feats.dim() < 2:
        raise ValueError('Feature tensor must have at least two dimensions')
    return feats.detach().float().reshape(-1, feats.shape[-1])


def avg_per_dim_var(feats: torch.Tensor) -> torch.Tensor:
    """Return the average variance across feature dimensions."""
    feats = flatten_feature_batch(feats)
    if feats.size(0) <= 1:
        return feats.new_zeros(())
    return feats.var(dim=0, unbiased=False).mean()


def mean_l0_norm(feats: torch.Tensor) -> torch.Tensor:
    """Return the mean number of non-zero features per sample."""
    feats = flatten_feature_batch(feats)
    return feats.ne(0).sum(dim=-1).float().mean()


def mean_l1_norm(feats: torch.Tensor) -> torch.Tensor:
    """Return the mean absolute feature mass per sample."""
    feats = flatten_feature_batch(feats)
    return feats.abs().sum(dim=-1).mean()


def mean_temporal_curvature(
    feats: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return mean ``1 - cosine`` curvature over consecutive latent triples.

    Inputs follow the world-model feature contract ``(B, T, ..., D)``. Any
    patch or spatial axes are flattened within each frame, so global and patch
    representations measure the direction change of the same whole-frame
    latent trajectory. This follows Ying Wang et al.'s Temporal Straightening
    diagnostic (arXiv:2603.12231). Sequences shorter than three frames have no
    curvature.
    """
    sequences = _time_feature_tensor(feats)
    if sequences.size(1) < 3:
        return sequences.new_zeros(())

    # 1. Consecutive latent differences are trajectory velocities. Flattening
    # only the axes after time preserves an independent curve for each example.
    velocities = sequences[:, 1:] - sequences[:, :-1]
    velocities = velocities.reshape(*velocities.shape[:2], -1)
    previous_velocity = velocities[:, :-1]
    current_velocity = velocities[:, 1:]

    # 2. Curvature is zero for parallel motion, one for an orthogonal turn, and
    # two for a reversal. A stationary segment has no defined turn, so map it to
    # zero instead of introducing an artificial curvature spike.
    speed_product = previous_velocity.norm(dim=-1) * current_velocity.norm(
        dim=-1
    )
    cosine = F.cosine_similarity(
        previous_velocity,
        current_velocity,
        dim=-1,
        eps=eps,
    )
    curvature = torch.where(
        speed_product > eps,
        1.0 - cosine.clamp(-1.0, 1.0),
        torch.zeros_like(cosine),
    )
    return curvature.mean()


def var_t(feats: torch.Tensor) -> torch.Tensor:
    """Return average per-dimension variance across time within each example."""
    sequences = _feature_time_sequences(feats)
    if sequences.size(1) <= 1:
        return sequences.new_zeros(())
    return sequences.var(dim=1, unbiased=False).mean()


def off_diag_cov_sum(feats: torch.Tensor) -> torch.Tensor:
    """Return squared off-diagonal covariance mass.

    A collapsed representation often has low variance. A tangled
    representation can also have large redundant covariance, so this metric
    helps separate "all dimensions died" from "dimensions are highly coupled".
    """
    with _autocast_disabled(feats):
        feats = flatten_feature_batch(feats).float()
        if feats.size(0) <= 1:
            return feats.new_zeros(())

        centered = feats - feats.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(feats.size(0) - 1, 1)
        off_diag = cov - torch.diag_embed(torch.diagonal(cov))
        return off_diag.square().sum()


def covariance_spectrum(feats: torch.Tensor) -> torch.Tensor:
    """Return covariance eigenvalues for rank-style diagnostics."""
    with _autocast_disabled(feats):
        feats = flatten_feature_batch(feats).float()
        if feats.size(0) <= 1:
            return feats.new_zeros(feats.shape[-1])

        centered = feats - feats.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(feats.size(0) - 1, 1)
        return torch.linalg.eigvalsh(cov.float()).clamp_min(0.0)


def effective_rank(feats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy rank of the covariance spectrum.

    This is useful for LeWM/PLDM collapse debugging: if a 192D latent only uses
    a tiny subspace, effective rank drops sharply even when MSE still decreases.
    """
    eigvals = covariance_spectrum(feats)
    total = eigvals.sum()
    probs = eigvals / (total + eps)
    entropy = -(probs * (probs + eps).log()).sum()
    return torch.where(total > eps, entropy.exp(), eigvals.new_zeros(()))


def participation_ratio(
    feats: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Second-moment rank proxy for the covariance spectrum."""
    eigvals = covariance_spectrum(feats)
    total = eigvals.sum()
    denom = eigvals.square().sum()
    ratio = total.square() / (denom + eps)
    return torch.where(
        (total > eps) & (denom > eps),
        ratio,
        eigvals.new_zeros(()),
    )


def feature_diagnostics(
    feats: torch.Tensor,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Build scalar diagnostics for one latent feature tensor."""
    with torch.no_grad():
        flat = flatten_feature_batch(feats)
        return {
            f'{prefix}_mean': flat.mean(),
            f'{prefix}_std': flat.std(unbiased=False),
            f'{prefix}_avg_per_dim_var': avg_per_dim_var(flat),
            f'{prefix}_off_diag_cov_sum': off_diag_cov_sum(flat),
            f'{prefix}_mean_l0_norm': mean_l0_norm(flat),
            f'{prefix}_mean_l1_norm': mean_l1_norm(flat),
            f'{prefix}_mean_temporal_curvature': mean_temporal_curvature(
                feats
            ),
            f'{prefix}_mean_l2_norm': flat.norm(dim=-1).mean(),
            f'{prefix}_matrix_fro_norm': flat.norm(),
            f'{prefix}_max_abs': flat.abs().max(),
            f'{prefix}_effective_rank': effective_rank(flat),
            f'{prefix}_participation_ratio': participation_ratio(flat),
        }


def _feature_time_sequences(feats: torch.Tensor) -> torch.Tensor:
    """Return features as ``(samples, T, D)`` for within-example time variance."""
    feats = _time_feature_tensor(feats)
    if feats.dim() == 3:
        return feats

    batch, time, *spatial, dim = feats.shape
    order = [0, *range(2, feats.dim() - 1), 1, feats.dim() - 1]
    return (
        feats.permute(order)
        .contiguous()
        .reshape(batch * _numel(spatial), time, dim)
    )


def _time_feature_tensor(feats: torch.Tensor) -> torch.Tensor:
    """Detach and normalize feature tensors to ``(B, T, ..., D)`` shape."""
    if feats.dim() < 2:
        raise ValueError('Feature tensor must have at least two dimensions')
    feats = feats.detach().float()
    if feats.dim() == 2:
        feats = feats.unsqueeze(1)
    return feats


def _numel(values: list[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def _autocast_disabled(tensor: torch.Tensor):
    """Disable autocast for diagnostics that require fp32 linear algebra."""
    device_type = tensor.device.type
    if device_type in {'cuda', 'cpu'}:
        return torch.amp.autocast(device_type=device_type, enabled=False)
    return nullcontext()


__all__ = [
    'avg_per_dim_var',
    'covariance_spectrum',
    'effective_rank',
    'feature_diagnostics',
    'flatten_feature_batch',
    'mean_l0_norm',
    'mean_l1_norm',
    'mean_temporal_curvature',
    'off_diag_cov_sum',
    'participation_ratio',
    'var_t',
]
