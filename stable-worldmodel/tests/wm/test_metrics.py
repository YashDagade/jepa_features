import torch

from stable_worldmodel.wm.metrics import (
    avg_per_dim_var,
    effective_rank,
    feature_diagnostics,
    flatten_feature_batch,
    mean_l0_norm,
    mean_l1_norm,
    mean_temporal_curvature,
    off_diag_cov_sum,
    var_t,
)


def test_flatten_feature_batch_keeps_last_axis_as_features():
    feats = torch.randn(2, 3, 4, 5)

    flat = flatten_feature_batch(feats)

    assert flat.shape == (24, 5)


def test_feature_diagnostics_are_finite_for_collapsed_features():
    feats = torch.zeros(4, 3, 8)

    diagnostics = feature_diagnostics(feats, 'encoder')

    assert diagnostics['encoder_avg_per_dim_var'].item() == 0
    assert diagnostics['encoder_effective_rank'].item() == 0
    assert diagnostics['encoder_mean_l0_norm'].item() == 0
    assert diagnostics['encoder_mean_l1_norm'].item() == 0
    assert diagnostics['encoder_mean_temporal_curvature'].item() == 0
    assert all(torch.isfinite(value) for value in diagnostics.values())


def test_l0_and_l1_norms_report_per_sample_feature_mass():
    feats = torch.tensor([[0.0, -2.0, 3.0], [4.0, 0.0, 0.0]])

    assert torch.allclose(mean_l0_norm(feats), torch.tensor(1.5))
    assert torch.allclose(mean_l1_norm(feats), torch.tensor(4.5))


def test_temporal_curvature_matches_one_minus_velocity_cosine():
    # 1. A straight path has zero curvature, an orthogonal turn has curvature
    # one, and reversing direction reaches the maximum value of two.
    straight = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    orthogonal = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    reversal = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])
    assert mean_temporal_curvature(straight).item() == 0.0
    assert mean_temporal_curvature(orthogonal).item() == 1.0
    assert mean_temporal_curvature(reversal).item() == 2.0
    assert (
        mean_temporal_curvature(torch.cat([straight, reversal])).item() == 1.0
    )

    # 2. A stationary segment has no direction and therefore no evidence of a
    # turn; it must not become an artificial maximum-curvature diagnostic.
    stationary = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]])
    assert mean_temporal_curvature(stationary).item() == 0.0


def test_temporal_curvature_flattens_patch_tokens_within_each_frame():
    global_curve = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    patch_curve = global_curve.reshape(1, 3, 2, 1)

    assert torch.allclose(
        mean_temporal_curvature(patch_curve),
        mean_temporal_curvature(global_curve),
    )


def test_variance_covariance_and_rank_are_positive_for_varied_features():
    feats = torch.randn(8, 3, 16)

    assert avg_per_dim_var(feats).item() > 0
    assert off_diag_cov_sum(feats).item() >= 0
    assert effective_rank(feats).item() > 1


def test_var_t_is_within_each_example_not_across_batch():
    feats = torch.tensor(
        [
            [[0.0], [2.0], [4.0]],
            [[7.0], [7.0], [7.0]],
        ]
    )

    assert torch.allclose(var_t(feats), torch.tensor(4.0 / 3.0))


def test_var_t_treats_patch_trajectories_independently():
    feats = torch.tensor(
        [
            [
                [[0.0], [7.0]],
                [[2.0], [7.0]],
                [[4.0], [7.0]],
            ],
        ]
    )

    assert torch.allclose(var_t(feats), torch.tensor(4.0 / 3.0))


def test_feature_diagnostics_do_not_log_var_t_by_default():
    feats = torch.randn(2, 3, 4)

    diagnostics = feature_diagnostics(feats, 'encoder')

    assert 'encoder_var_t' not in diagnostics


def test_rank_diagnostics_stay_float32_under_autocast():
    feats = torch.randn(8, 3, 16)

    with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):
        rank = effective_rank(feats)

    assert rank.dtype == torch.float32
    assert torch.isfinite(rank)
