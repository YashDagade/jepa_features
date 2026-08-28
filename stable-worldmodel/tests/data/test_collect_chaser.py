from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from scripts.data.collect_chaser import (
    TRAINING_KEYS,
    audit_hdf5,
    collect_condition,
    loader_exposure,
    run_collection,
    simulate_condition,
)
from stable_worldmodel.data import HDF5Dataset


def collection_cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            'seed': 111,
            'episodes': 3,
            'conditions': ['cue'],
            'overwrite': False,
            'calibration_approved': False,
            'dataset_cache_dir': str(tmp_path),
            'output_dir': str(tmp_path / 'runs'),
            'policy': {'action_noise': 0.0},
            'world': {
                'num_envs': 1,
                'cue_ids': [0, 1, 2],
                'max_episode_steps': 24,
                'runner_noise_std': 1.0,
                'image_shape': [224, 224],
            },
        }
    )


def test_full_collection_requires_explicit_calibration_approval(tmp_path):
    cfg = collection_cfg(tmp_path)

    with np.testing.assert_raises_regex(
        RuntimeError, 'calibration_approved=true'
    ):
        run_collection(cfg)


def test_collection_rejects_empty_or_duplicate_condition_lists(tmp_path):
    cfg = collection_cfg(tmp_path)
    cfg.calibration_approved = True

    for conditions, message in (
        ([], 'at least one'),
        (['cue', 'cue'], 'duplicates'),
    ):
        cfg.conditions = conditions
        with np.testing.assert_raises_regex(ValueError, message):
            run_collection(cfg)


def test_explicit_dataset_filename_requires_one_condition(tmp_path):
    cfg = collection_cfg(tmp_path)
    cfg.calibration_approved = True
    cfg.conditions = ['cue', 'reflection']
    cfg.dataset_filename = 'ambiguous.h5'

    with np.testing.assert_raises_regex(ValueError, 'exactly one'):
        run_collection(cfg)


def test_loader_exposure_matches_explicit_interval_semantics():
    events = np.zeros(20, dtype=bool)
    events[5] = True
    exposure = loader_exposure([events])

    # A length-20 episode has one legal clip. The event ending at row five is
    # in the first predicted interval and therefore in the clip union.
    assert exposure['legal_clips'] == 1
    assert exposure['predicted_intervals'] == 3
    assert exposure['event_intervals'] == 1
    assert exposure['event_clips'] == 1
    assert exposure['predicted_interval_exposure'] == 1 / 3
    assert exposure['event_clip_exposure'] == 1.0


def test_simulation_audit_has_exact_episode_lengths_and_balance_fields():
    audit, events = simulate_condition(
        'cue',
        episodes=12,
        episode_steps=24,
        seed=19,
        action_noise=0.0,
        runner_noise_std=1.0,
    )

    assert len(events) == 12
    assert {len(sequence) for sequence in events} == {24}
    assert audit['raw_frames'] == 12 * 24
    assert np.asarray(audit['cue_quadrant_counts']).shape == (3, 4)
    assert np.asarray(audit['cue_event_counts']).shape == (3,)
    assert audit['immediate_retriggers'] == 0
    assert set(audit['calibration']) == {
        'raw_event_rate',
        'predicted_interval_exposure',
        'event_clip_exposure',
        'no_immediate_retrigger',
        'no_boundary_contact',
        'orbit_direction_balance',
        'cue_quadrant_balance',
        'cue_episode_balance',
    }


def test_two_color_simulation_excludes_purple_and_compacts_audit_rows():
    audit, _ = simulate_condition(
        'cue',
        episodes=24,
        episode_steps=24,
        seed=19,
        action_noise=0.0,
        runner_noise_std=1.0,
        cue_ids=(0, 1),
    )

    assert audit['cue_ids'] == [0, 1]
    assert audit['cue_names'] == ['blue', 'green']
    assert np.asarray(audit['cue_quadrant_counts']).shape == (2, 4)
    assert np.asarray(audit['cue_event_counts']).shape == (2,)


def test_collection_reuses_hdf5_episode_and_training_contract(tmp_path):
    cfg = collection_cfg(tmp_path)
    path, manifest = collect_condition('cue', cfg, tmp_path / 'datasets')

    assert path.exists()
    assert manifest['episodes'] == 3
    assert manifest['raw_frames'] == 72
    assert len(manifest['sha256']) == 64
    with h5py.File(path, 'r') as data:
        np.testing.assert_array_equal(data['ep_len'][:], [24, 24, 24])
        np.testing.assert_array_equal(data['ep_offset'][:], [0, 24, 48])
        assert data['pixels'].shape == (72, 224, 224, 3)
        assert data['action'].shape == (72, 2)
        assert data['scenario_seed'][0] == 111
        assert {
            'event',
            'cue_id',
            'contact_quadrant',
            'pre_contact_red',
            'pre_contact_blue',
            'teleport_destination',
        }.issubset(data.keys())

    dataset = HDF5Dataset(
        path=path,
        keys_to_load=TRAINING_KEYS,
        frameskip=5,
        num_steps=4,
    )
    assert dataset.column_names == TRAINING_KEYS
    assert len(dataset) == 15
    item = dataset[0]
    assert item['pixels'].shape == (4, 3, 224, 224)
    assert item['action'].shape == (4, 10)
    assert set(item) == set(TRAINING_KEYS)


def test_collection_uses_explicit_two_color_filename_and_cues(tmp_path):
    cfg = collection_cfg(tmp_path)
    cfg.dataset_filename = 'chaser_cue_two_color.h5'
    cfg.world.cue_ids = [0, 1]
    path, manifest = collect_condition('cue', cfg, tmp_path / 'datasets')

    assert path.name == 'chaser_cue_two_color.h5'
    assert manifest['cue_ids'] == [0, 1]
    assert manifest['cue_names'] == ['blue', 'green']
    with h5py.File(path, 'r') as data:
        assert set(np.unique(data['cue_id'][:])) <= {0, 1}


def test_hdf5_audit_never_turns_absent_events_into_zeros(tmp_path):
    cfg = collection_cfg(tmp_path)
    path, original = collect_condition(
        'reflection', cfg, tmp_path / 'datasets'
    )
    audited = audit_hdf5(path)

    assert audited['events'] == original['events']
    assert audited['raw_event_rate'] == original['raw_event_rate']
    assert audited['inter_event_interval_mean'] is None
