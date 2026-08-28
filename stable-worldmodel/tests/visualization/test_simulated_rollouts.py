"""Tests for portal-conditioned simulated rollout visualization helpers."""

import gymnasium as gym
import numpy as np
import pytest
import torch

import stable_worldmodel  # noqa: F401 - registers custom environments
import stable_worldmodel.visualization.simulated_rollouts as rollout_viz
from stable_worldmodel.visualization.simulated_rollouts import (
    decode_next_prediction,
    near_portal_positions,
    ordered_hub_endpoints,
    portal_labels,
    preserve_module_modes,
    real_imagined_frame,
    save_simulated_rollout_panel,
)
from stable_worldmodel.wm.decoder import GlobalDecoder


def test_hidden_portals_are_ordered_top_to_bottom():
    # Hidden portals have no cue identity, so display order must be geometric.
    env = gym.make(
        'swm/TunnelRooms-v1',
        num_leaves=3,
        identity_mode='hidden',
        marker_style='filled18x18',
    ).unwrapped
    try:
        env.reset(seed=3072)
        endpoints = ordered_hub_endpoints(env)
        y = env.endpoint_positions[list(endpoints), 1]

        assert np.all(y[:-1] <= y[1:])
        assert portal_labels(env, endpoints) == ('top', 'middle', 'bottom')
    finally:
        env.close()


def test_visible_portals_keep_semantic_identity_order():
    # Shape-cued portals retain semantic endpoint identity rather than y-order.
    env = gym.make(
        'swm/TunnelRooms-v1',
        num_leaves=3,
        identity_mode='paired_shape',
        marker_style='shape_cues',
    ).unwrapped
    try:
        env.reset(seed=3072)
        endpoints = ordered_hub_endpoints(env)

        assert endpoints == (0, 1, 2)
        assert portal_labels(env, endpoints) == (
            'triangle',
            'diamond',
            'star',
        )
    finally:
        env.close()


def test_color_portals_apply_cue_assignment_before_labeling():
    # 1. Permute color cues while keeping the underlying geometry fixed.
    env = gym.make(
        'swm/TunnelRooms-v1',
        num_leaves=3,
        identity_mode='paired_color',
        marker_style='filled18x18',
    ).unwrapped
    try:
        # Initialize a valid randomized scenario before overriding only its cue map.
        env.reset(seed=3072)
        env.reset(
            seed=3072,
            options={
                'scenario': {
                    **env.scenario_spec(),
                    'cue_assignment': [1, 2, 0],
                }
            },
        )
        # 2. Order and labels must follow the assigned colors, not endpoint IDs.
        endpoints = ordered_hub_endpoints(env)

        assert endpoints == (2, 0, 1)
        assert portal_labels(env, endpoints) == (
            'green',
            'orange',
            'blue',
        )
    finally:
        env.close()


def test_near_portal_starts_are_valid_and_outside_trigger():
    # 1. Build deterministic approach points just outside each portal trigger.
    env = gym.make(
        'swm/TunnelRooms-v1',
        num_leaves=2,
        identity_mode='hidden',
        marker_style='filled18x18',
    ).unwrapped
    try:
        env.reset(seed=3072)
        endpoints = ordered_hub_endpoints(env)
        starts = near_portal_positions(env, endpoints, distance=15.0)

        # 2. Starts must remain in the hub and preserve the requested clearance.
        assert all(env.room_index(start) == 0 for start in starts)
        distances = np.linalg.norm(
            starts - env.endpoint_positions[list(endpoints)], axis=1
        )
        np.testing.assert_allclose(distances, 15.0)
        assert np.all(distances > env.trigger_radius)
    finally:
        env.close()


def test_real_imagined_frame_stacks_equal_images():
    real = np.zeros((28, 28, 3), dtype=np.uint8)
    imagined = np.full((28, 28, 3), 255, dtype=np.uint8)

    panel = real_imagined_frame(real, imagined)

    assert panel.shape == (60, 28, 3)
    assert np.all(panel[-1] == 255)


@pytest.mark.parametrize(
    ('labels', 'expected_views'),
    [
        (('green', 'orange'), 4),
        (('green', 'orange', 'blue'), 6),
    ],
)
def test_rollout_panel_keeps_only_portal_streams(
    monkeypatch, tmp_path, labels, expected_views
):
    # 1. Capture the TR2/TR3 layout without spending time encoding an MP4.
    captured = {}
    monkeypatch.setattr(
        rollout_viz,
        'save_panel_videos',
        lambda _path, panels, fps: captured.update(panels),
    )
    frame = np.zeros((28, 28, 3), dtype=np.uint8)

    # 2. Render one real/imagined stream pair for each available portal.
    output = save_simulated_rollout_panel(
        tmp_path,
        labels,
        [[frame] for _ in labels],
        [[frame] for _ in labels],
        fps=4,
    )

    # 3. No fixed-goal column remains: TR2 has four views and TR3 has six.
    assert tuple(captured) == tuple(f'{label} portal' for label in labels)
    assert 2 * len(captured) == expected_views
    assert output == tmp_path / 'env_0.mp4'


def test_preserve_module_modes_restores_mixed_submodule_state():
    # 1. Create a nested module with intentionally mixed train/eval state.
    module = torch.nn.Sequential(
        torch.nn.Linear(2, 2), torch.nn.Sequential(torch.nn.Linear(2, 2))
    )
    module.train()
    module[1].eval()
    before = [child.training for child in module.modules()]

    # 2. The context temporarily evaluates every nested module.
    with preserve_module_modes(module):
        assert not any(child.training for child in module.modules())

    # 3. Exit restores the exact per-module state rather than only the root.
    assert [child.training for child in module.modules()] == before


def test_preserve_module_modes_places_modules_on_requested_device():
    model = torch.nn.Linear(2, 2)
    decoder = torch.nn.Linear(2, 2)

    with preserve_module_modes(model, decoder, device='cpu'):
        assert next(model.parameters()).device.type == 'cpu'
        assert next(decoder.parameters()).device.type == 'cpu'

    assert next(model.parameters()).device.type == 'cpu'
    assert next(decoder.parameters()).device.type == 'cpu'


class _GlobalRolloutModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def rollout(self, info, actions):
        batch = actions.shape[0]
        return {'predicted_emb': torch.ones(batch, 1, 2, 8)}


def _rollout_inputs():
    info = {'pixels': torch.zeros(2, 1, 3, 14, 14)}
    actions = torch.zeros(2, 1, 2)
    return info, actions


def test_decode_next_prediction_supports_global_decoder():
    # Global rollout features route exclusively through GlobalDecoder.
    info, actions = _rollout_inputs()
    decoded = decode_next_prediction(
        _GlobalRolloutModel(),
        GlobalDecoder(
            global_dim=8,
            img_size=14,
            patch_size=7,
            dim=8,
            heads=2,
            depth=1,
        ),
        info,
        actions,
    )

    assert decoded.shape == (2, 14, 14, 3)
    assert decoded.dtype == np.uint8


def test_decode_next_prediction_rejects_extra_token_axis():
    class InvalidRolloutModel(_GlobalRolloutModel):
        def rollout(self, info, actions):
            batch = actions.shape[0]
            return {'predicted_emb': torch.ones(batch, 1, 2, 1, 8)}

    info, actions = _rollout_inputs()
    decoder = GlobalDecoder(
        global_dim=8,
        img_size=14,
        patch_size=7,
        dim=8,
        heads=2,
        depth=1,
    )

    with pytest.raises(ValueError, match=r'requires \(B, D\)'):
        decode_next_prediction(InvalidRolloutModel(), decoder, info, actions)
