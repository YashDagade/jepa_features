import gymnasium as gym
import numpy as np
import pytest

import stable_worldmodel  # noqa: F401  # Registers Gym environments.
from stable_worldmodel.envs.tunnel_rooms import (
    TunnelRoomsEnv,
    TunnelSearchPolicy,
    VisibleRoutingPolicy,
    VonMisesCoveragePolicy,
)
from stable_worldmodel.envs.tunnel_rooms.marker_raster import (
    NEUTRAL_RGB,
    marker_mask,
    marker_size,
)
from stable_worldmodel.envs.tunnel_rooms.routing import (
    certify_ranked_route,
    nearest_unexplored_order,
)
from stable_worldmodel.envs.tunnel_rooms.rendering import endpoint_colors
from stable_worldmodel.world.env_pool import EnvPool


def scenario(
    pairing: int = 0,
    target_leaf: int = 0,
    identity_mode: str = 'hidden',
    speed: float = 20.0,
):
    target = [190, 60] if target_leaf == 0 else [190, 160]
    return {
        'identity_mode': identity_mode,
        'pairing': pairing,
        'endpoint_positions': [[70, 60], [70, 160], [158, 60], [158, 160]],
        'agent_position': [35, 110],
        'target_position': target,
        'target_leaf': target_leaf,
        'trigger_radius': 10,
        'landing_offset': 14,
        'rearm_distance': 28,
        'speed': speed,
    }


def paired_shape_scenario(pairing=0, target_leaf=0):
    target_y = (46, 112, 178)[target_leaf]
    return {
        'identity_mode': 'paired_shape',
        'marker_style': 'shape_cues',
        'marker_width': 20,
        'marker_height': 20,
        'pairing': pairing,
        'endpoint_positions': [
            [60, 112],
            [82, 160],
            [56, 46],
            [158, 46],
            [158, 112],
            [158, 178],
        ],
        'agent_position': [28, 112],
        'target_position': [190, target_y],
        'target_leaf': target_leaf,
        'trigger_radius': 10,
        'landing_offset': 14,
        'rearm_distance': 28,
        'speed': 20,
        'num_leaves': 3,
    }


def reset_scenario(**kwargs):
    env = TunnelRoomsEnv(identity_mode=kwargs.get('identity_mode', 'hidden'))
    observation, info = env.reset(options={'scenario': scenario(**kwargs)})
    return env, observation, info


def rollout(env, policy, scenario_value, max_steps=180):
    observation, info = env.reset(options={'scenario': scenario_value})
    policy.set_env(env)
    for step in range(max_steps):
        input_info = {**info, **observation, 'step_idx': np.int32(step)}
        action = policy.get_action(input_info)
        observation, _, terminated, _, info = env.step(action)
        if terminated:
            return True
    return False


def one_leaf_scenario(speed: float = 20.0):
    return {
        'num_leaves': 1,
        'identity_mode': 'hidden',
        'marker_style': 'filled18x18',
        'marker_width': 18,
        'marker_height': 18,
        'agent_render_radius': 9,
        'pairing': 0,
        'endpoint_positions': [[70, 112], [158, 112]],
        'agent_position': [35, 70],
        'target_position': [190, 160],
        'target_leaf': 0,
        'trigger_radius': 10,
        'landing_offset': 14,
        'rearm_distance': 28,
        'speed': speed,
    }


def test_one_leaf_control_has_one_certified_route_and_no_leaf_divider():
    env = TunnelRoomsEnv(num_leaves=1, marker_style='filled18x18')
    env.reset(
        seed=17, options={'variation_values': {'agent.render_radius': [9]}}
    )
    spec = env.scenario_spec()
    route = certify_ranked_route(spec)

    assert spec['layout'] == 'hub_left_1_leaves_right'
    assert np.asarray(spec['endpoint_positions']).shape == (2, 2)
    assert spec['pairing'] == 0
    assert spec['target_leaf'] == 0
    assert route.search_order == (0,)
    assert route.correct_hub_endpoint == 0
    assert route.target_rank == 1
    assert env.render()[112, 180].tolist() != [24, 26, 28]


def test_one_leaf_hidden_expert_solves_the_direct_navigation_task():
    env = TunnelRoomsEnv(num_leaves=1, marker_style='filled18x18')
    scenario_value = one_leaf_scenario()

    assert rollout(env, TunnelSearchPolicy(seed=7), scenario_value)
    assert env.audit_metadata()['hub_entry_trace'] == [0]


def test_registration_and_public_contract():
    env = gym.make('swm/TunnelRooms-v1')
    observation, info = env.reset(seed=17)
    assert env.unwrapped.speed == 5.0
    assert set(observation) == {'proprio', 'state'}
    assert observation['proprio'].shape == (2,)
    assert observation['state'].dtype == np.float32
    assert set(info) == {
        'env_name',
        'goal',
        'goal_state',
        'goal_proprio',
        'pos_agent',
        'pos_target',
        'distance_to_target',
    }
    assert info['goal'].shape == (224, 224, 3)
    assert info['goal'].dtype == np.uint8
    assert env.render().shape == (224, 224, 3)


def test_reset_is_deterministic_and_goal_is_rendered_target():
    env = TunnelRoomsEnv()
    observation_a, info_a = env.reset(seed=108)
    spec_a = env.scenario_spec()
    observation_b, info_b = env.reset(seed=108)
    assert spec_a == env.scenario_spec()
    np.testing.assert_array_equal(
        observation_a['state'], observation_b['state']
    )
    np.testing.assert_array_equal(info_a['goal'], info_b['goal'])
    np.testing.assert_array_equal(
        info_b['goal'], env._render_frame(info_b['goal_state'])
    )


def test_walls_collide_and_target_terminates():
    env, _, _ = reset_scenario()
    env._set_state(np.array([98, 110], dtype=np.float32))
    observation, _, terminated, _, _ = env.step(np.array([1, 0]))
    assert observation['state'][0] <= 98
    assert not terminated
    env._set_state(env.target_position)
    _, _, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert terminated


@pytest.mark.parametrize(
    ('pairing', 'expected_landing'),
    [(0, [172, 60]), (1, [172, 160])],
)
def test_swept_activation_teleports_to_correct_pair(pairing, expected_landing):
    env, _, _ = reset_scenario(pairing=pairing)
    env._set_state(np.array([45, 60], dtype=np.float32))
    observation, _, _, _, _ = env.step(np.array([1, 0], dtype=np.float32))
    np.testing.assert_allclose(observation['state'], expected_landing)
    assert env.teleport_count == 1


def test_arrival_is_locked_then_rearms_for_bidirectional_return():
    env, _, _ = reset_scenario(pairing=0)
    env._set_state(np.array([45, 60], dtype=np.float32))
    env.step(np.array([1, 0], dtype=np.float32))
    observation, _, _, _, _ = env.step(np.array([-1, 0], dtype=np.float32))
    assert env.teleport_count == 1
    assert observation['state'][0] > 112

    env.step(np.array([1, 0], dtype=np.float32))
    env.step(np.array([1, 0], dtype=np.float32))
    env.step(np.array([-1, 0], dtype=np.float32))
    observation, _, _, _, _ = env.step(np.array([-1, 0], dtype=np.float32))
    assert env.teleport_count == 2
    np.testing.assert_allclose(observation['state'], [56, 60])


def test_hidden_rendering_masks_pairing_but_paired_color_reveals_it():
    env = TunnelRoomsEnv()
    env.reset(
        options={'scenario': scenario(pairing=0, identity_mode='hidden')}
    )
    hidden_a = env.render()
    hidden_goal_a = env._get_info()['goal']
    env.reset(
        options={'scenario': scenario(pairing=1, identity_mode='hidden')}
    )
    np.testing.assert_array_equal(hidden_a, env.render())
    np.testing.assert_array_equal(hidden_goal_a, env._get_info()['goal'])

    env.reset(
        options={'scenario': scenario(pairing=0, identity_mode='paired_color')}
    )
    visible_a = env.render()
    env.reset(
        options={'scenario': scenario(pairing=1, identity_mode='paired_color')}
    )
    assert not np.array_equal(visible_a, env.render())


def test_cue_assignment_is_independent_of_routing_and_legacy_defaults():
    env = TunnelRoomsEnv(identity_mode='paired_color')
    base = scenario(pairing=1, identity_mode='paired_color')

    env.reset(options={'scenario': base})
    assert 'cue_assignment' not in env.scenario_spec()
    assert 'tunnel_colors' not in env.scenario_spec()
    assert 'tunnel_color_names' not in env.scenario_spec()
    assert env.cue_assignment == (0, 1)
    legacy_colors = [env._marker_color(index) for index in range(4)]

    env.reset(options={'scenario': {**base, 'cue_assignment': [1, 0]}})
    spec = env.scenario_spec()
    assert spec['pairing'] == 1
    assert spec['cue_assignment'] == [1, 0]
    assert not np.array_equal(legacy_colors[0], env._marker_color(0))
    replay_colors = endpoint_colors(
        'paired_color', 2, spec['pairing'], spec['cue_assignment']
    )
    for hub in env.hub_endpoint_indices():
        paired = env._paired_endpoint(hub)
        np.testing.assert_array_equal(
            env._marker_color(hub), env._marker_color(paired)
        )
    for index, color in enumerate(replay_colors):
        np.testing.assert_array_equal(color, env._marker_color(index))

    hidden = TunnelRoomsEnv(identity_mode='hidden')
    hidden.reset(options={'scenario': {**base, 'identity_mode': 'hidden'}})
    image_a = hidden.render()
    hidden.reset(
        options={
            'scenario': {
                **base,
                'identity_mode': 'hidden',
                'cue_assignment': [1, 0],
            }
        }
    )
    np.testing.assert_array_equal(image_a, hidden.render())


def test_blue_green_palette_replays_without_a_new_identity_mode():
    # 1. Reuse paired-color routing while replacing only the second cue RGB.
    env = TunnelRoomsEnv(identity_mode='paired_color')
    blue_green = {
        **scenario(identity_mode='paired_color'),
        'tunnel_colors': [[40, 153, 78], [83, 98, 216]],
        'tunnel_color_names': ['green', 'blue'],
    }
    env.reset(options={'scenario': blue_green})

    # 2. Preserve green, replace orange by blue, and keep the agent red.
    np.testing.assert_array_equal(env._marker_color(0), [40, 153, 78])
    np.testing.assert_array_equal(env._marker_color(1), [83, 98, 216])
    frame = env.render()
    assert np.any(np.all(frame == [83, 98, 216], axis=-1))
    assert np.any(np.all(frame == [213, 38, 45], axis=-1))

    # 3. Persist the custom palette so later scenario replay is pixel-identical.
    replay = env.scenario_spec()
    assert replay['identity_mode'] == 1
    assert replay['tunnel_colors'] == blue_green['tunnel_colors']
    assert replay['tunnel_color_names'] == blue_green['tunnel_color_names']
    env.reset(options={'scenario': replay})
    np.testing.assert_array_equal(frame, env.render())


def test_sampled_hub_routes_do_not_cross_the_other_trigger_zone():
    env = TunnelRoomsEnv()
    for seed in range(10):
        env.reset(seed=seed)
        points = env.endpoint_positions
        start = env.scenario_spec()['agent_position']
        assert np.linalg.norm(points[0] - points[1]) >= env.min_hub_separation
        for intended, other in ((0, 1), (1, 0)):
            assert (
                env._segment_distance(
                    np.asarray(start), points[intended], points[other]
                )
                > env.trigger_radius
            )


def test_three_leaf_sampling_is_deterministic_and_naturally_certified():
    env = TunnelRoomsEnv(num_leaves=3)
    env.reset(seed=41021)
    expected = env.scenario_spec()
    env.reset(seed=41021)
    spec = env.scenario_spec()

    assert spec == expected
    assert spec['num_leaves'] == 3
    assert len(spec['endpoint_positions']) == 6
    assert sorted(spec['search_order']) == [0, 1, 2]
    assert spec['target_rank'] in (1, 2, 3)
    assert (
        spec['search_order'][spec['target_rank'] - 1]
        == spec['correct_hub_endpoint']
    )
    assert 'ordered_pair' not in spec


def test_four_leaf_sampling_is_deterministic_and_uses_four_unique_colors():
    env = TunnelRoomsEnv(
        identity_mode='paired_color',
        num_leaves=4,
        marker_style='filled18x18',
    )
    env.reset(seed=41021)
    expected = env.scenario_spec()
    env.reset(seed=41021)
    spec = env.scenario_spec()

    assert spec == expected
    assert spec['num_leaves'] == 4
    assert len(spec['endpoint_positions']) == 8
    assert sorted(spec['search_order']) == [0, 1, 2, 3]
    assert (
        spec['search_order'][spec['target_rank'] - 1]
        == spec['correct_hub_endpoint']
    )
    assert spec['tunnel_color_names'] == ['green', 'orange', 'blue', 'magenta']
    assert len(spec['tunnel_colors']) == 4
    assert len({tuple(env._marker_color(index)) for index in range(4)}) == 4


def test_four_leaf_sampling_reaches_every_rank_across_fixed_seeds():
    env = TunnelRoomsEnv(num_leaves=4)
    observed = set()
    for seed in range(64):
        env.reset(seed=seed)
        observed.add(env.scenario_spec()['target_rank'])

    assert observed == {1, 2, 3, 4}


@pytest.mark.parametrize(
    'option',
    [
        'ordered_pair_index',
        'scenario_index',
        'scenario_seed_base',
        'target_rank',
        'target_rank_probs',
    ],
)
def test_three_leaf_sampling_rejects_retired_rank_forcing_options(option):
    env = TunnelRoomsEnv(num_leaves=3)
    with pytest.raises(ValueError, match='no longer accepts'):
        env.reset(seed=41021, options={option: 0})


def test_three_leaf_geometry_randomizes_roles_without_clipping():
    env = TunnelRoomsEnv(num_leaves=3)
    env.reset(seed=7)
    anchor_labels = set()
    assignments = set()
    for _ in range(180):
        sampled = env._sample_three_leaf_scenario()
        hubs = sampled['tunnel.endpoints'][:3]
        start = sampled['agent.position']
        anchor = int(np.argmin(np.linalg.norm(hubs - start, axis=1)))
        anchor_labels.add(anchor)
        assignments.add((sampled['target.leaf'], sampled['tunnel.pairing']))

        y_distances = sorted(
            abs(float(hubs[idx, 1] - hubs[anchor, 1]))
            for idx in range(3)
            if idx != anchor
        )
        assert 43.0 <= y_distances[0] <= 56.0
        assert 84.0 <= y_distances[1] <= 104.0
        assert np.all((38.0 <= hubs[:, 1]) & (hubs[:, 1] <= 186.0))
        assert not np.any(np.isin(hubs[:, 1], [38.0, 186.0]))

    assert anchor_labels == {0, 1, 2}
    assert assignments == {
        (target_leaf, pairing)
        for target_leaf in range(3)
        for pairing in range(6)
    }


def test_nearest_unexplored_order_uses_return_landings():
    initial = np.array([0.0, 0.0], dtype=np.float32)
    hubs = np.array([[1.0, 0.0], [4.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    returns = np.array([[9.0, 0.0], [4.0, 0.0], [10.0, 0.0]], dtype=np.float32)

    assert nearest_unexplored_order(initial, hubs, returns) == [0, 2, 1]


def test_replay_recertifies_one_based_target_rank():
    env = TunnelRoomsEnv(num_leaves=3)
    env.reset(seed=41021)
    base = env.scenario_spec()
    permutation = env._pairing_permutations[base['pairing']]

    for expected_rank, hub_endpoint in enumerate(
        base['search_order'], start=1
    ):
        target_leaf = int(permutation[hub_endpoint])
        target_room = target_leaf + 1
        bounds = env._room_bounds(target_room)
        leaf_endpoint = np.asarray(base['endpoint_positions'][3 + target_leaf])
        corners = np.array(
            [
                [bounds[0], bounds[2]],
                [bounds[0], bounds[3]],
                [bounds[1], bounds[2]],
                [bounds[1], bounds[3]],
            ],
            dtype=np.float32,
        )
        target = corners[
            np.argmax(np.linalg.norm(corners - leaf_endpoint, axis=1))
        ]
        replay = {
            **base,
            'target_leaf': target_leaf,
            'target_position': target.tolist(),
            'target_rank': 99,
            'search_order': list(reversed(base['search_order'])),
        }

        env.reset(options={'scenario': replay})
        certified = env.scenario_spec()
        assert certified['target_rank'] == expected_rank
        assert certified['search_order'] == base['search_order']


@pytest.mark.parametrize('rank', [1, 2, 3])
def test_three_leaf_search_policy_solves_every_realized_rank(rank):
    template_env = TunnelRoomsEnv(num_leaves=3)
    template_env.reset(seed=41021)
    base = template_env.scenario_spec()
    correct_hub = base['search_order'][rank - 1]
    target_leaf = int(
        template_env._pairing_permutations[base['pairing']][correct_hub]
    )
    bounds = template_env._room_bounds(target_leaf + 1)
    replay = {
        **base,
        'speed': 20.0,
        'target_leaf': target_leaf,
        'target_position': [190.0, float((bounds[2] + bounds[3]) / 2)],
    }

    env = TunnelRoomsEnv(identity_mode='hidden', num_leaves=3)
    assert rollout(env, TunnelSearchPolicy(seed=0), replay, max_steps=120)

    color_replay = {**replay, 'identity_mode': 'paired_color'}
    color_env = TunnelRoomsEnv(identity_mode='paired_color', num_leaves=3)
    assert rollout(
        color_env,
        VisibleRoutingPolicy(seed=0),
        color_replay,
        max_steps=120,
    )


def test_three_leaf_pairing_is_bidirectional_for_all_permutations():
    env = TunnelRoomsEnv(num_leaves=3)
    base = {
        'identity_mode': 'hidden',
        'endpoint_positions': [
            [60, 112],
            [82, 160],
            [56, 46],
            [158, 46],
            [158, 112],
            [158, 178],
        ],
        'agent_position': [28, 112],
        'target_position': [190, 46],
        'target_leaf': 0,
        'trigger_radius': 10,
        'landing_offset': 14,
        'rearm_distance': 28,
        'speed': 20,
        'num_leaves': 3,
    }
    for pairing in range(6):
        scenario_value = {**base, 'pairing': pairing}
        env.reset(options={'scenario': scenario_value})
        for hub_endpoint in env.hub_endpoint_indices():
            leaf_endpoint = env._paired_endpoint(hub_endpoint)
            assert env._paired_endpoint(leaf_endpoint) == hub_endpoint


def test_configured_minimum_hub_separation_rejects_close_tunnels():
    env = TunnelRoomsEnv()
    invalid = scenario()
    invalid['endpoint_positions'][1] = [70, 90]
    with pytest.raises(ValueError, match='minimum separation'):
        env.reset(options={'scenario': invalid})


def test_validation_rejects_landing_offset_at_rearm_distance():
    env = TunnelRoomsEnv()
    invalid = scenario()
    invalid['landing_offset'] = invalid['rearm_distance']
    with pytest.raises(ValueError, match='landing_offset'):
        env.reset(options={'scenario': invalid})


def test_marker_stroke_width_defaults_to_thin_but_is_configurable():
    env = TunnelRoomsEnv()
    fixed = scenario(identity_mode='paired_color')
    env.reset(options={'scenario': fixed})
    thin = env.render()
    env.reset(
        options={
            'scenario': fixed,
            'variation_values': {
                'tunnel.marker_stroke_width': np.array([4], np.float32)
            },
        }
    )
    bold = env.render()
    tunnel_color = np.array([40, 153, 78], dtype=np.uint8)
    thin_pixels = np.all(thin == tunnel_color, axis=-1).sum()
    bold_pixels = np.all(bold == tunnel_color, axis=-1).sum()
    # Pair A renders two integer-centered 15x15 markers with a 13x13 hole.
    assert thin_pixels == 2 * (15**2 - 13**2)
    assert bold_pixels > thin_pixels
    assert (
        env.variation_space['tunnel']['marker_stroke_width'].init_value[0] == 1
    )
    np.testing.assert_array_equal(
        env.endpoint_positions, fixed['endpoint_positions']
    )


def test_legacy_scenario_defaults_to_byte_identical_outline_contract():
    legacy = scenario(identity_mode='paired_color')
    env = TunnelRoomsEnv(marker_style='filled18x18')
    _, legacy_info = env.reset(options={'scenario': legacy})
    legacy_render = env.render()

    explicit = {
        **legacy,
        'marker_style': 'outline',
        'marker_width': 15,
        'marker_height': 15,
        'marker_size': 7.0,
        'marker_stroke_width': 1.0,
    }
    _, explicit_info = env.reset(options={'scenario': explicit})

    assert env.marker_style == 'outline'
    assert env.scenario_spec()['marker_width'] == 15
    assert env.scenario_spec()['marker_height'] == 15
    np.testing.assert_array_equal(legacy_render, env.render())
    np.testing.assert_array_equal(legacy_info['goal'], explicit_info['goal'])


def test_filled18x18_renders_exact_filled_squares_and_round_trips():
    filled = {
        **scenario(identity_mode='paired_color'),
        'marker_style': 'filled18x18',
        'marker_width': 18,
        'marker_height': 18,
    }
    env = TunnelRoomsEnv()
    _, info = env.reset(options={'scenario': filled})
    image = env.render()
    spec = env.scenario_spec()

    assert spec['marker_style'] == 'filled18x18'
    assert spec['marker_width'] == 18
    assert spec['marker_height'] == 18
    for endpoint_idx, (x, y) in enumerate(filled['endpoint_positions']):
        color = env._marker_color(endpoint_idx)
        x_start = int(x) - 9
        y_start = int(y) - 9
        marker = image[y_start : y_start + 18, x_start : x_start + 18]
        assert marker.shape == (18, 18, 3)
        assert np.all(marker == color)

    pair_a = np.array([40, 153, 78], dtype=np.uint8)
    pair_b = np.array([232, 142, 34], dtype=np.uint8)
    assert np.all(image == pair_a, axis=-1).sum() == 2 * 18**2
    assert np.all(image == pair_b, axis=-1).sum() == 2 * 18**2

    replay = TunnelRoomsEnv()
    _, replay_info = replay.reset(options={'scenario': spec})
    np.testing.assert_array_equal(image, replay.render())
    np.testing.assert_array_equal(info['goal'], replay_info['goal'])


def test_paired_shape_renders_inverse_matched_masks_for_all_permutations():
    env = TunnelRoomsEnv(
        identity_mode='paired_shape',
        marker_style='shape_cues',
        num_leaves=3,
    )
    for pairing in range(6):
        env.reset(options={'scenario': paired_shape_scenario(pairing)})
        image = env.render()
        spec = env.scenario_spec()
        assert spec['marker_style'] == 'shape_cues'
        assert (spec['marker_width'], spec['marker_height']) == (20, 20)
        for hub_endpoint in env.hub_endpoint_indices():
            expected = marker_mask(hub_endpoint)
            width, height = marker_size(hub_endpoint)
            leaf_endpoint = env._paired_endpoint(hub_endpoint)
            assert env._paired_endpoint(leaf_endpoint) == hub_endpoint
            for endpoint in (hub_endpoint, leaf_endpoint):
                x, y = env.endpoint_positions[endpoint].astype(int)
                patch = image[
                    y - height // 2 : y - height // 2 + height,
                    x - width // 2 : x - width // 2 + width,
                ]
                np.testing.assert_array_equal(
                    np.all(patch == NEUTRAL_RGB, axis=-1), expected
                )


def test_paired_shape_uses_exact_neutral_rgb_not_pair_colors():
    env = TunnelRoomsEnv(
        identity_mode='paired_shape',
        marker_style='shape_cues',
        num_leaves=3,
    )
    env.reset(options={'scenario': paired_shape_scenario()})
    image = env.render()
    neutral_pixels = np.all(image == [62, 94, 112], axis=-1)

    assert neutral_pixels.sum() == 2 * sum(
        int(marker_mask(identity).sum()) for identity in range(3)
    )
    for pair_color in ([40, 153, 78], [232, 142, 34], [83, 98, 216]):
        assert not np.any(np.all(image == pair_color, axis=-1))


def test_paired_shape_goal_and_current_frames_draw_agent_last():
    env = TunnelRoomsEnv(
        identity_mode='paired_shape',
        marker_style='shape_cues',
        num_leaves=3,
    )
    env.reset(options={'scenario': paired_shape_scenario()})
    hub_endpoint = 0
    leaf_endpoint = env._paired_endpoint(hub_endpoint)

    env._set_state(env.endpoint_positions[hub_endpoint])
    current = env.render()
    env._set_goal_state(env.endpoint_positions[leaf_endpoint])
    goal = env._get_info()['goal']

    agent_rgb = np.array([213, 38, 45], dtype=np.uint8)
    for image, endpoint in (
        (current, hub_endpoint),
        (goal, leaf_endpoint),
    ):
        x, y = env.endpoint_positions[endpoint].astype(int)
        np.testing.assert_array_equal(image[y, x], agent_rgb)


@pytest.mark.parametrize(
    ('identity_mode', 'marker_style'),
    [
        ('paired_shape', 'outline'),
        ('paired_shape', 'filled18x18'),
        (2, 1),
    ],
)
def test_paired_shape_constructor_rejects_non_shape_cue_markers(
    identity_mode, marker_style
):
    with pytest.raises(ValueError, match='paired_shape.*shape_cues'):
        TunnelRoomsEnv(
            identity_mode=identity_mode,
            marker_style=marker_style,
        )


def test_paired_shape_scenario_rejects_legacy_outline_default():
    invalid = scenario(identity_mode='paired_shape')
    with pytest.raises(ValueError, match='paired_shape.*shape_cues'):
        TunnelRoomsEnv().reset(options={'scenario': invalid})


def test_paired_shape_reset_override_rejects_outline_immediately():
    env = TunnelRoomsEnv()
    with pytest.raises(ValueError, match='paired_shape.*shape_cues'):
        env.reset(
            seed=41021,
            options={'variation_values': {'tunnel.identity_mode': 2}},
        )


def test_agent_render_radius_changes_pixels_not_physics():
    base = scenario(identity_mode='paired_color')
    regular = TunnelRoomsEnv()
    larger = TunnelRoomsEnv()
    regular.reset(options={'scenario': base})
    larger.reset(options={'scenario': {**base, 'agent_render_radius': 9.0}})

    assert regular.agent_radius == larger.agent_radius == 7.0
    assert regular.agent_render_radius == 7.0
    assert larger.agent_render_radius == 9.0
    np.testing.assert_array_equal(
        regular._room_bounds(0), larger._room_bounds(0)
    )
    assert np.count_nonzero(
        np.all(larger.render() == [213, 38, 45], axis=-1)
    ) > np.count_nonzero(np.all(regular.render() == [213, 38, 45], axis=-1))
    assert larger.scenario_spec()['agent_render_radius'] == 9.0


@pytest.mark.parametrize(('width', 'height'), [(17, 18), (18, 17), (19, 19)])
def test_filled18x18_rejects_other_dimensions(width, height):
    invalid = {
        **scenario(),
        'marker_style': 'filled18x18',
        'marker_width': width,
        'marker_height': height,
    }
    with pytest.raises(ValueError, match='requires marker_width=18'):
        TunnelRoomsEnv().reset(options={'scenario': invalid})


def test_reset_marker_appearance_mapping_is_serialized_in_scenario():
    appearance = {
        'marker_style': 'filled18x18',
        'marker_width': 18,
        'marker_height': 18,
    }
    env = TunnelRoomsEnv()
    env.reset(seed=41021, options={'marker_appearance': appearance})
    spec = env.scenario_spec()

    assert {key: spec[key] for key in appearance} == appearance


def test_marker_appearance_does_not_change_seeded_scenario_geometry():
    outline = TunnelRoomsEnv(num_leaves=3)
    filled = TunnelRoomsEnv(num_leaves=3)
    outline.reset(seed=41021)
    filled.reset(
        seed=41021,
        options={
            'marker_appearance': {
                'marker_style': 'filled18x18',
                'marker_width': 18,
                'marker_height': 18,
            }
        },
    )
    outline_spec = outline.scenario_spec()
    filled_spec = filled.scenario_spec()

    for key in (
        'agent_position',
        'endpoint_positions',
        'pairing',
        'target_leaf',
        'target_position',
        'search_order',
        'target_rank',
    ):
        np.testing.assert_array_equal(outline_spec[key], filled_spec[key])


def test_scenario_latents_are_only_available_through_audit_accessor():
    env, observation, info = reset_scenario(pairing=1)
    public_values = set(observation) | set(info)
    assert 'pairing' not in public_values
    assert 'endpoint_positions' not in public_values
    assert 'target_rank' not in public_values
    assert 'search_order' not in public_values
    assert 'hub_entry_trace' not in public_values
    assert env.scenario_spec()['pairing'] == 1


def test_hub_entry_trace_is_audit_only():
    env, _, _ = reset_scenario(pairing=0)
    env._set_state(np.array([45, 60], dtype=np.float32))
    observation, _, _, _, info = env.step(np.array([1, 0], dtype=np.float32))

    assert env.audit_metadata()['hub_entry_trace'] == [0]
    assert 'hub_entry_trace' not in observation
    assert 'hub_entry_trace' not in info


@pytest.mark.parametrize(
    ('pairing', 'target_leaf'),
    [(0, 0), (0, 1), (1, 0), (1, 1)],
)
def test_hidden_search_policy_succeeds_after_observed_outcomes(
    pairing, target_leaf
):
    env = TunnelRoomsEnv(identity_mode='hidden')
    success = rollout(
        env,
        TunnelSearchPolicy(seed=0),
        scenario(pairing=pairing, target_leaf=target_leaf, speed=5),
    )
    assert success


@pytest.mark.parametrize(
    ('pairing', 'target_leaf'),
    [(0, 0), (0, 1), (1, 0), (1, 1)],
)
def test_visible_policy_routes_directly_to_matching_tunnel(
    pairing, target_leaf
):
    env = TunnelRoomsEnv(identity_mode='paired_color')
    success = rollout(
        env,
        VisibleRoutingPolicy(seed=0),
        scenario(
            pairing=pairing,
            target_leaf=target_leaf,
            identity_mode='paired_color',
            speed=5,
        ),
    )
    assert success
    assert env.teleport_count == 1


def test_search_policy_uses_visible_cues_without_a_second_policy_class():
    env = TunnelRoomsEnv(identity_mode='paired_color')
    success = rollout(
        env,
        TunnelSearchPolicy(seed=0),
        scenario(
            pairing=1,
            target_leaf=0,
            identity_mode='paired_color',
            speed=5,
        ),
    )
    assert success
    assert env.teleport_count == 1


@pytest.mark.parametrize('pairing', range(6))
def test_visible_policy_routes_by_paired_shape_for_all_permutations(pairing):
    env = TunnelRoomsEnv(
        identity_mode='paired_shape',
        marker_style='shape_cues',
        num_leaves=3,
    )
    success = rollout(
        env,
        VisibleRoutingPolicy(seed=0),
        paired_shape_scenario(pairing, target_leaf=pairing % 3),
        max_steps=120,
    )
    assert success
    assert env.teleport_count == 1


def test_visible_policy_raises_when_pairing_is_hidden():
    env, observation, info = reset_scenario(identity_mode='hidden')
    policy = VisibleRoutingPolicy(seed=0)
    policy.set_env(env)
    with pytest.raises(RuntimeError, match='paired_color mode'):
        policy.get_action({**info, **observation, 'step_idx': np.int32(0)})


def test_search_policy_action_repeat_resets_at_episode_start():
    env, observation, info = reset_scenario(speed=5)
    policy = TunnelSearchPolicy(action_repeat_prob=1.0, seed=0)
    policy.set_env(env)

    first = policy.get_action({**info, **observation, 'step_idx': np.int32(0)})
    repeated = policy.get_action(
        {
            **info,
            **observation,
            'state': np.array([190, 160], dtype=np.float32),
            'goal_state': np.array([180, 160], dtype=np.float32),
            'step_idx': np.int32(1),
        }
    )
    np.testing.assert_allclose(repeated, first)

    policy.reset()
    reset_action = policy.get_action(
        {
            **info,
            **observation,
            'state': np.array([190, 160], dtype=np.float32),
            'goal_state': np.array([180, 160], dtype=np.float32),
            'step_idx': np.int32(2),
        }
    )
    assert not np.allclose(reset_action, first)


def test_search_and_coverage_policy_resets_are_mask_isolated():
    pool = EnvPool([TunnelRoomsEnv, TunnelRoomsEnv])
    try:
        search = TunnelSearchPolicy(action_repeat_prob=1.0, seed=0)
        search.set_env(pool)
        search._tested[0].add(0)
        search._tested[1].add(1)
        search._candidate[:] = [0, 1]
        search._return_ready[:] = [True, True]
        search._last_action = np.ones((2, 2), dtype=np.float32)
        search._last_action_valid[:] = True

        search.reset(mask=np.array([True, False]))

        assert search._tested == [set(), {1}]
        assert search._candidate == [None, 1]
        assert search._return_ready == [False, True]
        assert search._last_action_valid.tolist() == [False, True]

        coverage = VonMisesCoveragePolicy(seed=0)
        coverage.set_env(pool)
        coverage._directions[:] = [0.25, 0.75]
        coverage.reset(mask=np.array([True, False]))
        assert np.isnan(coverage._directions[0])
        assert coverage._directions[1] == np.float32(0.75)
    finally:
        pool.close()


def test_search_policy_records_all_failed_tunnels():
    env = TunnelRoomsEnv()
    policy = TunnelSearchPolicy()
    policy.set_env(env)
    policy._remember_failed(0, 0)
    policy._remember_failed(0, 1)
    assert policy._tested[0] == {0, 1}


def test_von_mises_coverage_policy_has_paper_aligned_step_bound():
    env, observation, info = reset_scenario(speed=5)
    policy = VonMisesCoveragePolicy(seed=14)
    policy.set_env(env)
    for step in range(20):
        action = policy.get_action(
            {**info, **observation, 'step_idx': np.int32(step)}
        )
        assert np.linalg.norm(action * env.speed) <= 2.45 + 1e-6
        observation, _, _, _, info = env.step(action)


def test_von_mises_coverage_policy_does_not_depend_on_hidden_pairing():
    env_a, obs_a, info_a = reset_scenario(pairing=0)
    env_b, obs_b, info_b = reset_scenario(pairing=1)
    policy_a = VonMisesCoveragePolicy(seed=25)
    policy_b = VonMisesCoveragePolicy(seed=25)
    policy_a.set_env(env_a)
    policy_b.set_env(env_b)
    action_a = policy_a.get_action({**info_a, **obs_a, 'step_idx': 0})
    action_b = policy_b.get_action({**info_b, **obs_b, 'step_idx': 0})
    np.testing.assert_array_equal(action_a, action_b)
