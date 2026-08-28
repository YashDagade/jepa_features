import gymnasium as gym
import numpy as np
import pytest

import stable_worldmodel as swm  # Registers Gym environments.
from stable_worldmodel.envs.chaser import ChaserEnv, ChaserExpertPolicy
from stable_worldmodel.envs.chaser.env import (
    ARENA_CENTER,
    CONTACT_DISTANCE,
    CUE_COLORS,
)


def replay_scenario(cue_id=0, orbit_direction=1):
    return {
        'scenario_seed': 19,
        'red_position': [170.0, 112.0],
        'blue_position': [54.0, 112.0],
        'cue_id': cue_id,
        'orbit_direction': orbit_direction,
        'orbit_radius': 58.0,
        'orbit_phase': np.pi,
    }


def test_registration_and_public_contract():
    env = gym.make('swm/Chaser-v0')
    observation, info = env.reset(seed=17)

    assert set(observation) == {'proprio', 'state'}
    assert observation['proprio'].shape == (2,)
    assert observation['state'].shape == (4,)
    assert observation['state'].dtype == np.float32
    assert env.action_space.shape == (2,)
    assert info['cue_id'].dtype == np.int8
    assert info['event'].dtype == np.bool_
    assert info['runner_noise_seed'].dtype == np.uint32
    assert info['runner_noise_std'].dtype == np.float32
    assert env.render().shape == (224, 224, 3)
    assert env.render().dtype == np.uint8


def test_reset_and_replay_are_deterministic():
    env = ChaserEnv(mapping_mode='cue')
    observation_a, info_a = env.reset(seed=91)
    frame_a = env.render()
    trajectory_a = []
    for _ in range(8):
        observation, _, _, _, info = env.step(np.array([0.4, -0.2]))
        trajectory_a.append(observation['state'])

    observation_b, info_b = env.reset(seed=91)
    np.testing.assert_array_equal(
        observation_a['state'], observation_b['state']
    )
    np.testing.assert_array_equal(frame_a, env.render())
    assert int(info_a['cue_id']) == int(info_b['cue_id'])
    trajectory_b = [
        env.step(np.array([0.4, -0.2]))[0]['state'] for _ in range(8)
    ]
    np.testing.assert_allclose(trajectory_a, trajectory_b)


def test_counterfactual_cues_change_only_the_target_color():
    env = ChaserEnv(mapping_mode='cue')
    scenario = replay_scenario(cue_id=0)
    env.reset(options={'scenario': scenario})
    blue = env.render()
    state = env._get_obs()['state']

    env.reset(options={'scenario': {**scenario, 'cue_id': 1}})
    green = env.render()

    np.testing.assert_array_equal(state, env._get_obs()['state'])
    difference = np.any(blue != green, axis=-1)
    assert 300 < difference.sum() < 500
    y, x = np.rint(env.blue_position).astype(int)[::-1]
    assert difference[y - 7 : y + 8, x - 7 : x + 8].any()


def test_all_three_target_colors_are_distinct_at_training_resolution():
    env = ChaserEnv(mapping_mode='cue')
    crops = []
    for cue_id in range(len(CUE_COLORS)):
        env.reset(options={'scenario': replay_scenario(cue_id=cue_id)})
        frame = env.render()
        x, y = np.rint(env.blue_position).astype(int)
        crops.append(frame[y - 7 : y + 8, x - 7 : x + 8])

    for left in range(len(crops)):
        for right in range(left + 1, len(crops)):
            assert not np.array_equal(crops[left], crops[right])


def test_two_color_variant_samples_only_blue_and_green():
    env = ChaserEnv(mapping_mode='cue', cue_ids=(0, 1))
    observed = {int(env.reset(seed=seed)[1]['cue_id']) for seed in range(64)}

    assert observed == {0, 1}
    with pytest.raises(ValueError, match='cue_id must be'):
        env.reset(options={'scenario': replay_scenario(cue_id=2)})


@pytest.mark.parametrize('cue_ids', [(), (0, 0), (0, 3)])
def test_cue_subset_rejects_empty_duplicate_or_unknown_ids(cue_ids):
    with pytest.raises(ValueError, match='cue_ids'):
        ChaserEnv(mapping_mode='cue', cue_ids=cue_ids)


def test_reflection_mapping_uses_the_opposite_ray():
    env = ChaserEnv(mapping_mode='reflection')
    env.reset(seed=1)
    contact = np.array([192.0, 112.0], dtype=np.float32)
    destination = env._teleport_destination(contact)

    np.testing.assert_allclose(destination, [32.0, 112.0], atol=1e-5)
    np.testing.assert_allclose(
        np.linalg.norm(destination - ARENA_CENTER),
        np.linalg.norm(contact - ARENA_CENTER),
    )


@pytest.mark.parametrize(
    ('cue_id', 'expected_quadrants'),
    [(0, [2, 3, 0, 1]), (1, [1, 2, 3, 0]), (2, [3, 0, 1, 2])],
)
def test_color_cue_rotates_quadrants_and_preserves_radius(
    cue_id, expected_quadrants
):
    env = ChaserEnv(mapping_mode='cue')
    env.reset(options={'scenario': replay_scenario(cue_id=cue_id)})
    contacts = np.array(
        [[72.0, 72.0], [152.0, 72.0], [152.0, 152.0], [72.0, 152.0]],
        dtype=np.float32,
    )

    for contact, expected_quadrant in zip(contacts, expected_quadrants):
        destination = env._teleport_destination(contact)
        assert env._quadrant(destination) == expected_quadrant
        np.testing.assert_allclose(
            np.linalg.norm(destination - ARENA_CENTER),
            np.linalg.norm(contact - ARENA_CENTER),
        )


def test_all_three_cues_produce_distinct_counterfactual_destinations():
    env = ChaserEnv(mapping_mode='cue')
    contact = np.array([192.0, 112.0], dtype=np.float32)
    destinations = []
    for cue_id in range(len(CUE_COLORS)):
        env.reset(options={'scenario': replay_scenario(cue_id=cue_id)})
        destinations.append(env._teleport_destination(contact))

    assert len({tuple(destination) for destination in destinations}) == 3


def test_swept_contact_detects_a_between_frame_crossing():
    fraction = ChaserEnv._swept_contact_fraction(
        np.array([-30.0, 0.0]),
        np.array([30.0, 0.0]),
        CONTACT_DISTANCE,
    )

    assert fraction is not None
    assert 0.0 < fraction < 0.5
    assert (
        ChaserEnv._swept_contact_fraction(
            np.array([-30.0, 30.0]),
            np.array([30.0, 30.0]),
            CONTACT_DISTANCE,
        )
        is None
    )


def test_contact_teleports_once_and_records_exact_metadata():
    # Isolate swept-contact bookkeeping from the independent runner-noise test.
    env = ChaserEnv(mapping_mode='cue', runner_noise_std=0.0)
    scenario = {
        **replay_scenario(cue_id=0),
        'red_position': [179.0, 112.0],
        'blue_position': [192.0, 112.0],
        'orbit_radius': 80.0,
        'orbit_phase': 0.0,
    }
    env.reset(options={'scenario': scenario})
    observation, _, _, _, info = env.step(np.array([1.0, 0.0]))

    assert info['event']
    assert info['event_count'] == 1
    assert info['contact_quadrant'] == 2
    assert info['destination_quadrant'] == env._quadrant(
        info['teleport_destination']
    )
    np.testing.assert_array_equal(
        observation['proprio'], info['teleport_destination']
    )
    np.testing.assert_array_equal(info['pre_contact_red'], [179.0, 112.0])
    np.testing.assert_array_equal(info['pre_contact_blue'], [192.0, 112.0])

    _, _, _, _, next_info = env.step(np.array([1.0, 0.0]))
    assert not next_info['event']
    assert next_info['event_count'] == 1


def test_blue_motion_is_smooth_contained_and_does_not_wrap():
    env = ChaserEnv(mapping_mode='reflection')
    env.reset(seed=31)
    positions = [env.blue_position.copy()]
    for _ in range(200):
        env.step(np.zeros(2, dtype=np.float32))
        positions.append(env.blue_position.copy())
    positions = np.asarray(positions)
    velocity = np.diff(positions, axis=0)

    assert positions.min() >= 25.0
    assert positions.max() <= 199.0
    np.testing.assert_allclose(
        np.linalg.norm(velocity, axis=1), 3.0, atol=1e-4
    )
    assert np.max(np.linalg.norm(np.diff(velocity, axis=0), axis=1)) < 1.0


def test_runner_noise_seed_changes_the_smooth_target_path():
    env = ChaserEnv(mapping_mode='reflection')
    scenario = {**replay_scenario(cue_id=-1), 'mapping_mode': 'reflection'}
    paths = []
    for noise_seed in (7, 11):
        env.reset(
            options={'scenario': {**scenario, 'runner_noise_seed': noise_seed}}
        )
        paths.append(
            np.asarray(
                [env.step(np.zeros(2))[4]['pos_blue'] for _ in range(30)]
            )
        )

    assert not np.allclose(paths[0], paths[1])


def test_runner_noise_strength_is_configurable_and_replay_safe():
    scenario = {**replay_scenario(cue_id=-1), 'mapping_mode': 'reflection'}
    paths = []
    for noise_std in (0.0, 1.0):
        env = ChaserEnv(mapping_mode='reflection', runner_noise_std=noise_std)
        env.reset(
            options={
                'scenario': {
                    **scenario,
                    'runner_noise_seed': 7,
                    'runner_noise_std': noise_std,
                }
            }
        )
        paths.append(
            np.asarray(
                [env.step(np.zeros(2))[4]['pos_blue'] for _ in range(30)]
            )
        )

    assert not np.allclose(paths[0], paths[1])


def test_nonzero_red_actions_keep_constant_speed():
    env = ChaserEnv(mapping_mode='reflection')
    scenario = {**replay_scenario(), 'cue_id': -1}
    env.reset(options={'scenario': scenario})
    start = env.red_position.copy()

    env.step(np.array([0.25, 0.75], dtype=np.float32))

    np.testing.assert_allclose(
        np.linalg.norm(env.red_position - start), env.red_speed, atol=1e-5
    )


def test_reset_balances_orbit_directions_over_many_seeds():
    env = ChaserEnv(mapping_mode='reflection')
    directions = []
    for seed in range(400):
        _, info = env.reset(seed=seed)
        directions.append(int(info['orbit_direction']))
    clockwise = sum(direction < 0 for direction in directions)
    assert abs(clockwise - len(directions) / 2) < 0.1 * len(directions)


def test_expert_supports_scalar_and_vectorized_world_contracts():
    env = ChaserEnv()
    _, info = env.reset(seed=9)
    policy = ChaserExpertPolicy(seed=9)
    policy.set_env(env)
    action = policy.get_action(info)
    direction = info['pos_blue'] - info['pos_red']

    assert action.shape == (2,)
    np.testing.assert_allclose(action, direction / np.linalg.norm(direction))

    world = swm.World(
        'swm/Chaser-v0',
        num_envs=2,
        add_pixels=False,
        max_episode_steps=8,
    )
    try:
        vector_policy = ChaserExpertPolicy(seed=9)
        world.set_policy(vector_policy)
        world.reset(seed=9)
        assert vector_policy.get_action(world.infos).shape == (2, 2)
    finally:
        world.close()
