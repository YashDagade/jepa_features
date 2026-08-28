from pathlib import Path

import h5py
import numpy as np
import torch

from stable_worldmodel.probes.chaser import (
    ChaserProbeDataset,
    build_chaser_probe_samples,
    fit_chaser_probes,
)
from stable_worldmodel.probes.location import FrozenLocationFeatures


class _DummyWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.encoder = torch.nn.Linear(1, 1)
        self.projector = torch.nn.Identity()
        self.action_encoder = torch.nn.Identity()

    def predict(self, embedding, _action):
        return embedding


def _write_chaser(
    path: Path, episodes: int = 60, length: int = 40, cue_count: int = 3
) -> None:
    rows = episodes * length
    offsets = np.arange(episodes, dtype=np.int64) * length
    episode_ids = np.repeat(10_000 + np.arange(episodes), length)
    cue = np.repeat(np.arange(episodes) % cue_count, length).astype(np.int8)
    event = np.zeros(rows, dtype=bool)
    contact = np.full(rows, -1, dtype=np.int8)
    destination = np.full(rows, -1, dtype=np.int8)
    position = np.zeros((rows, 2), dtype=np.float32)
    runner = np.zeros((rows, 2), dtype=np.float32)
    rotations = (2, 1, 3)

    # 1. Give every episode one valid first contact and smooth position labels.
    for episode, offset in enumerate(offsets):
        steps = np.arange(length, dtype=np.float32)
        position[offset : offset + length] = np.stack(
            (steps + episode, 2 * steps + episode), axis=1
        )
        runner[offset : offset + length] = np.stack(
            (3 * steps + episode, steps - episode), axis=1
        )
        event_row = offset + 20
        quadrant = (episode // cue_count) % 4
        event[event_row] = True
        contact[event_row] = quadrant
        destination[event_row] = (
            quadrant + rotations[episode % cue_count]
        ) % 4

    # 2. Store the minimal production schema used by the frozen probe.
    with h5py.File(path, 'w') as h5:
        h5.create_dataset('pixels', data=np.zeros((rows, 8, 8, 3), np.uint8))
        h5.create_dataset('action', data=np.ones((rows, 2), np.float32))
        h5.create_dataset('pos_red', data=position)
        h5.create_dataset('pos_blue', data=runner)
        h5.create_dataset('id', data=episode_ids)
        h5.create_dataset('cue_id', data=cue)
        h5.create_dataset('event', data=event)
        h5.create_dataset('contact_quadrant', data=contact)
        h5.create_dataset('destination_quadrant', data=destination)
        h5.create_dataset('ep_offset', data=offsets)
        h5.create_dataset(
            'ep_len', data=np.full(episodes, length, dtype=np.int64)
        )


def test_chaser_samples_are_deterministic_balanced_and_episode_disjoint(
    tmp_path,
):
    source = tmp_path / 'chaser.h5'
    _write_chaser(source)

    first = build_chaser_probe_samples(
        source, seed=7, frameskip=2, history_size=3, samples_per_split=16
    )
    second = build_chaser_probe_samples(
        source, seed=7, frameskip=2, history_size=3, samples_per_split=16
    )

    # 1. The immutable selection and its digest must reproduce exactly.
    assert first.sample_sha256 == second.sample_sha256
    assert np.array_equal(first.frame_rows, second.frame_rows)

    # 2. Every split has balanced event-imminence labels and distinct episodes.
    episode_sets = []
    for split in range(3):
        split_mask = first.split_codes == split
        event_mask = split_mask & first.event_balanced_mask
        assert first.event_imminent[event_mask].mean() == 0.5
        assert first.first_contact_mask[split_mask].any()
        episode_sets.append(set(first.episode_ids[split_mask]))
    assert not (episode_sets[0] & episode_sets[1])
    assert not (episode_sets[0] & episode_sets[2])
    assert not (episode_sets[1] & episode_sets[2])

    # 3. Dataset reads the same aligned action contract as location probes.
    item = ChaserProbeDataset(first)[0]
    assert item['pixels'].shape == (3, 8, 8, 3)
    assert item['action_history'].shape == (3, 4)


def test_representation_features_keep_encoder_projector_predictor_separate():
    extractor = FrozenLocationFeatures(
        _DummyWorldModel(), image_size=4, freeze_model=False
    )
    encoder = torch.arange(2 * 1 * 4, dtype=torch.float32).reshape(2, 1, 4)
    projector = encoder + 10
    predictor = encoder + 20
    extractor._encode_stage_history = lambda pixels, action: (
        encoder,
        projector,
        predictor,
        4,
    )

    readout = extractor.representation_features(
        torch.zeros(2, 1, 4, 4, 3), torch.zeros(2, 1, 1)
    )

    assert readout.shape == (2, 3, 4)
    assert torch.equal(readout[:, 0], encoder[:, 0])
    assert torch.equal(readout[:, 1], projector[:, 0])
    assert torch.equal(readout[:, 2], predictor[:, 0])


def test_chaser_probes_recover_linearly_encoded_motion_and_rare_rule(tmp_path):
    source = tmp_path / 'chaser.h5'
    _write_chaser(source)
    samples = build_chaser_probe_samples(
        source, seed=11, frameskip=2, history_size=3, samples_per_split=64
    )
    count = len(samples.frame_rows)
    features = np.zeros((count, 3, 20), dtype=np.float32)
    sample_rows = np.arange(count)

    # 1. Encode the shared discrete tasks into all representation stages.
    for stage in range(3):
        features[sample_rows, stage, samples.cue_id] = 1
        features[:, stage, 3] = samples.event_imminent
        contacts = samples.first_contact_mask
        contact_rows = np.flatnonzero(contacts)
        features[
            contact_rows,
            stage,
            4 + samples.contact_quadrant[contact_rows],
        ] = 1
        features[
            contact_rows,
            stage,
            8 + samples.destination_quadrant[contact_rows],
        ] = 1

    # 2. Align current positions to current stages and future positions to prediction.
    current = np.concatenate(
        (samples.red_current, samples.runner_current), axis=1
    )
    future = np.concatenate(
        (samples.red_future, samples.runner_future), axis=1
    )
    features[:, 0, 12:16] = current
    features[:, 1, 12:16] = current
    features[:, 2, 12:16] = future

    result = fit_chaser_probes(features, samples, ridge_alpha=1e-6)

    # 3. Every task should be recovered when its signal is explicitly linear.
    for stage in result['stages'].values():
        assert stage['cue']['balanced_accuracy'] > 0.99
        assert stage['event_imminence']['balanced_accuracy'] > 0.99
        assert stage['contact_quadrant']['balanced_accuracy'] > 0.99
        assert stage['destination_quadrant']['balanced_accuracy'] > 0.99
        assert stage['motion_position']['average']['mae'] < 1e-3


def test_two_color_probe_uses_binary_chance_level(tmp_path):
    source = tmp_path / 'chaser_two_color.h5'
    _write_chaser(source, episodes=80, cue_count=2)
    samples = build_chaser_probe_samples(
        source, seed=13, frameskip=2, history_size=3, samples_per_split=64
    )
    features = np.zeros((len(samples.frame_rows), 3, 20), dtype=np.float32)

    for stage in range(3):
        features[np.arange(len(samples.cue_id)), stage, samples.cue_id] = 1
    result = fit_chaser_probes(features, samples, ridge_alpha=1e-6)

    assert samples.summary()['cue_values'] == [0, 1]
    for stage in result['stages'].values():
        assert stage['cue']['chance_balanced_accuracy'] == 0.5
        assert stage['cue']['balanced_accuracy'] > 0.99
