import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from stable_worldmodel.probes.location import (
    FEATURE_READOUT,
    FeatureCacheMetadata,
    FrozenLocationFeatures,
    LocationProbeSidecarDataset,
    extract_location_features,
    fit_location_probe,
    fit_tunnel_objective_probes,
    flatten_probe_metrics,
    flatten_probe_summary,
    load_feature_cache,
    regression_metrics,
    sample_sha256,
    snapshot_cache_path,
    write_feature_cache,
)


class _DummyWorldModel(torch.nn.Module):
    """Track mode and gradient flags without implementing WM computation."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.encoder = torch.nn.Linear(1, 1)
        self.projector = torch.nn.Identity()
        self.action_encoder = torch.nn.Identity()

    def predict(self, embedding, _action):
        return embedding


TARGET_NAMES = (
    'agent_current_x',
    'agent_current_y',
    'agent_future_x',
    'agent_future_y',
    'tunnel_green_hub_x',
    'tunnel_green_hub_y',
    'tunnel_green_leaf_x',
    'tunnel_green_leaf_y',
)


def test_frozen_location_features_preserves_live_model_state_when_opted_out():
    model = _DummyWorldModel().train()

    FrozenLocationFeatures(model, freeze_model=False)

    assert model.training
    assert model.weight.requires_grad


def test_frozen_location_features_freezes_standalone_model_by_default():
    model = _DummyWorldModel().train()

    FrozenLocationFeatures(model)

    assert not model.training
    assert not model.weight.requires_grad


def test_location_readout_preserves_global_feature_vector():
    # 1. Supply global vectors whose values expose any accidental reduction.
    extractor = FrozenLocationFeatures(
        _DummyWorldModel(), image_size=4, freeze_model=False
    )
    embedding = torch.arange(2 * 1 * 7, dtype=torch.float32).reshape(2, 1, 7)
    prediction = embedding + 100
    extractor._encode_history = lambda pixels, action: (
        embedding,
        prediction,
        4,
    )

    # 2. Read out current and predicted features without adding a token axis.
    readout = extractor.location_features(
        torch.zeros(2, 1, 4, 4, 3), torch.zeros(2, 1, 1)
    )

    # 3. Global mode must preserve the selected vectors exactly.
    assert torch.equal(readout[:, 0], embedding[:, 0, :4])
    assert torch.equal(readout[:, 1], prediction[:, 0, :4])


def test_dominant_objective_probe_recovers_nearest_tunnel_geometry():
    rng = np.random.default_rng(7)
    count = 180
    splits = np.repeat(np.arange(3, dtype=np.int8), count // 3)
    agent = rng.uniform([30, 20], [190, 204], size=(count, 2))
    hubs = np.broadcast_to(
        np.asarray([[112, 45], [112, 179]], np.float32), (count, 2, 2)
    )
    target_names = (
        'agent_current_x',
        'agent_current_y',
        'agent_future_x',
        'agent_future_y',
        'tunnel_top_hub_x',
        'tunnel_top_hub_y',
        'tunnel_bottom_hub_x',
        'tunnel_bottom_hub_y',
    )
    targets = np.concatenate((agent, agent, hubs.reshape(count, -1)), axis=1)
    displacement = hubs - agent[:, None]
    distances = np.linalg.norm(displacement, axis=-1)
    identity = distances.argmin(axis=1)
    nearest = displacement[np.arange(count), identity]
    nearest_distance = distances[np.arange(count), identity]
    direction = nearest / nearest_distance[:, None]

    # Encode all three dominant-objective targets directly in both stages.
    stage = np.column_stack((identity, nearest_distance, direction))
    features = np.stack((stage, stage), axis=1)
    result = fit_tunnel_objective_probes(
        features, targets, splits, target_names, ridge_alpha=1e-6
    )

    for metrics in result.values():
        assert metrics['nearest_tunnel_identity']['balanced_accuracy'] > 0.99
        assert metrics['nearest_tunnel_distance']['mae'] < 1e-3
        assert metrics['nearest_tunnel_direction']['mean_cosine'] > 0.999


def test_tunnel_probe_recovers_green_to_positional_slot_binding():
    count = 180
    splits = np.repeat(np.arange(3, dtype=np.int8), count // 3)
    color_slot = np.arange(count, dtype=np.int64) % 2
    agent = np.tile(np.asarray([[100.0, 112.0]]), (count, 1))
    hubs = np.broadcast_to(
        np.asarray([[112.0, 45.0], [112.0, 179.0]]), (count, 2, 2)
    )
    green = hubs[np.arange(count), color_slot]
    target_names = (
        'agent_current_x',
        'agent_current_y',
        'agent_future_x',
        'agent_future_y',
        'tunnel_green_hub_x',
        'tunnel_green_hub_y',
        'tunnel_top_hub_x',
        'tunnel_top_hub_y',
        'tunnel_bottom_hub_x',
        'tunnel_bottom_hub_y',
    )
    targets = np.concatenate(
        (agent, agent, green, hubs.reshape(count, -1)), axis=1
    )
    stage = np.column_stack((color_slot, np.ones(count), np.ones(count)))
    features = np.stack((stage, stage), axis=1)

    result = fit_tunnel_objective_probes(
        features, targets, splits, target_names, ridge_alpha=1e-6
    )

    for metrics in result.values():
        assert metrics['color_binding']['balanced_accuracy'] > 0.99


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_source(path: Path) -> np.ndarray:
    # Build three fixed-length episodes with one non-finite terminal action.
    rows = 36
    action = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
    action[-1] = np.nan
    position = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2) / 10
    with h5py.File(path, 'w') as h5:
        h5.create_dataset('pixels', data=np.zeros((rows, 4, 4, 3), np.uint8))
        h5.create_dataset('goal', data=np.ones((rows, 4, 4, 3), np.uint8))
        h5.create_dataset('action', data=action)
        h5.create_dataset('pos_agent', data=position)
        h5.create_dataset('ep_idx', data=np.repeat(np.arange(3), 12))
    return action


def _write_sidecar(
    path: Path,
    source: Path,
    *,
    episode_ids: np.ndarray | None = None,
    future_rows: np.ndarray | None = None,
    num_preds: int = 1,
    source_sha256: str | None = None,
    schema_version: int = 1,
) -> None:
    # 1. Select one history/future sample from each source episode.
    sample_ids = np.asarray([101, 102, 103], np.int64)
    episodes = (
        np.asarray([0, 1, 2], np.int64)
        if episode_ids is None
        else np.asarray(episode_ids, np.int64)
    )
    frame_rows = np.asarray([[2, 4], [14, 16], [26, 28]], np.int64)
    futures = (
        frame_rows[:, -1] + num_preds * 2
        if future_rows is None
        else np.asarray(future_rows, np.int64)
    )
    splits = np.asarray([0, 1, 2], np.int8)
    # 2. Derive targets from the exact source rows referenced by the sidecar.
    with h5py.File(source, 'r') as source_h5:
        position = source_h5['pos_agent'][:]
    targets = np.zeros((3, len(TARGET_NAMES)), np.float32)
    targets[:, :2] = position[frame_rows[:, -1]]
    targets[:, 2:4] = position[futures]
    targets[:, 4:] = np.asarray([10.0, 20.0, 30.0, 40.0])
    # 3. Bind sample identity, splits, row provenance, and targets in one digest.
    digest = sample_sha256(
        sample_ids,
        episodes,
        frame_rows,
        futures,
        splits,
        targets,
        TARGET_NAMES,
    )
    # 4. Persist both source-file provenance and sample-level provenance.
    with h5py.File(path, 'w') as h5:
        h5.attrs['schema_version'] = schema_version
        h5.attrs['source_hdf5'] = str(source)
        h5.attrs['source_sha256'] = (
            _sha256_file(source) if source_sha256 is None else source_sha256
        )
        h5.attrs['sample_sha256'] = digest
        h5.attrs['frameskip'] = 2
        h5.attrs['num_preds'] = num_preds
        h5.attrs['history_size'] = 2
        h5.attrs['target_names_json'] = json.dumps(TARGET_NAMES)
        h5.create_dataset('sample_id', data=sample_ids)
        h5.create_dataset('episode_id', data=episodes)
        h5.create_dataset('split', data=splits)
        h5.create_dataset('frame_rows', data=frame_rows)
        h5.create_dataset('future_row', data=futures)
        h5.create_dataset('target', data=targets)


@pytest.mark.parametrize('schema_version', [1, 2])
def test_sidecar_normalizes_like_training_and_keeps_final_action(
    tmp_path, schema_version
):
    # 1. Materialize a source/sidecar pair for each supported schema version.
    source = tmp_path / 'source.h5'
    sidecar = tmp_path / 'sidecar.h5'
    action = _write_source(source)
    _write_sidecar(sidecar, source, schema_version=schema_version)

    # 2. Load the sidecar and recover the same statistics used during training.
    dataset = LocationProbeSidecarDataset(sidecar)
    item = dataset[0]
    finite_action = action[np.isfinite(action).all(axis=1)]

    # 3. Exclude the non-finite action, but keep the final valid history action.
    assert torch.allclose(
        dataset.action_input_mean,
        torch.from_numpy(finite_action.mean(axis=0)),
    )
    assert torch.allclose(
        dataset.action_input_std,
        torch.from_numpy(finite_action.std(axis=0)),
    )
    assert torch.equal(
        item['action_history'][-1], torch.from_numpy(action[4:6].reshape(-1))
    )
    assert 'goal' not in item


def test_sidecar_rejects_episode_overlap_between_splits(tmp_path):
    source = tmp_path / 'source.h5'
    sidecar = tmp_path / 'sidecar.h5'
    _write_source(source)
    _write_sidecar(sidecar, source, episode_ids=np.asarray([0, 0, 2]))

    with pytest.raises(ValueError, match='appears in both'):
        LocationProbeSidecarDataset(sidecar)


def test_sidecar_rejects_misaligned_future_row(tmp_path):
    source = tmp_path / 'source.h5'
    sidecar = tmp_path / 'sidecar.h5'
    _write_source(source)
    _write_sidecar(sidecar, source, future_rows=np.asarray([7, 18, 30]))

    with pytest.raises(ValueError, match=r'num_preds \* frameskip'):
        LocationProbeSidecarDataset(sidecar)


def test_sidecar_rejects_source_content_not_matching_hash(tmp_path):
    # 1. Write a valid provenance pair, then mutate the source in place.
    source = tmp_path / 'source.h5'
    sidecar = tmp_path / 'sidecar.h5'
    _write_source(source)
    _write_sidecar(sidecar, source)
    with h5py.File(source, 'r+') as source_h5:
        source_h5['pixels'][0, 0, 0, 0] = 1

    # 2. The sidecar must fail before features are extracted from changed data.
    with pytest.raises(ValueError, match='source HDF5 SHA-256'):
        LocationProbeSidecarDataset(sidecar)


def test_sidecar_rejects_multiple_prediction_chunks(tmp_path):
    source = tmp_path / 'source.h5'
    sidecar = tmp_path / 'sidecar.h5'
    _write_source(source)
    _write_sidecar(sidecar, source, num_preds=2)

    with pytest.raises(ValueError, match='num_preds == 1'):
        LocationProbeSidecarDataset(sidecar)


class _FeatureDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            'pixels': torch.full((2, 1), float(index)),
            'action_history': torch.zeros(2, 1),
            'target': torch.tensor([index, index + 1.0]),
            'split': torch.tensor(index % 3, dtype=torch.int8),
            'sample_id': torch.tensor(index),
        }


class _MixedStateExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 3), torch.nn.ReLU()
        )
        self.seen_inference_mode = []
        self.seen_final_action = []

    def location_features(self, pixels, action_history):
        self.seen_inference_mode.append(torch.is_inference_mode_enabled())
        self.seen_final_action.append(action_history[:, -1].clone())
        current = self.model(pixels[:, -1].float())
        return torch.stack((current, current + 1), dim=1)


class _RngConsumingExtractor(torch.nn.Module):
    def location_features(self, pixels, action_history):
        return torch.rand((pixels.shape[0], 2, 3), device=pixels.device)


class _AutocastObservingExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enabled = []
        self.dtypes = []

    def location_features(self, pixels, action_history):
        self.enabled.append(torch.is_autocast_enabled(pixels.device.type))
        self.dtypes.append(torch.get_autocast_dtype(pixels.device.type))
        current = pixels[:, -1].float()
        return torch.stack((current, current), dim=1)


def test_feature_extraction_restores_nested_state_and_never_builds_gradients():
    # 1. Begin with intentionally mixed train/eval state across nested modules.
    extractor = _MixedStateExtractor()
    extractor.train()
    extractor.model.eval()
    prior_requires_grad = [
        parameter.requires_grad for parameter in extractor.parameters()
    ]

    # 2. Extract all probe inputs through the production batching path.
    features, targets, splits, sample_ids = extract_location_features(
        _FeatureDataset(), extractor, batch_size=2
    )

    # 3. Verify outputs, inference mode, and exact restoration of module state.
    assert features.shape == (4, 2, 3)
    assert targets.shape == (4, 2)
    assert splits.shape == (4,)
    assert sample_ids.tolist() == [0, 1, 2, 3]
    assert extractor.training
    assert not extractor.model.training
    assert all(extractor.seen_inference_mode)
    assert all(parameter.grad is None for parameter in extractor.parameters())
    assert [
        parameter.requires_grad for parameter in extractor.parameters()
    ] == prior_requires_grad


@pytest.mark.parametrize(
    ('autocast_dtype', 'expected_enabled'),
    [(None, False), (torch.bfloat16, True)],
)
def test_feature_extraction_uses_explicit_cpu_autocast(
    autocast_dtype, expected_enabled
):
    extractor = _AutocastObservingExtractor()

    extract_location_features(
        _FeatureDataset(),
        extractor,
        batch_size=2,
        autocast_dtype=autocast_dtype,
    )

    assert extractor.enabled == [expected_enabled, expected_enabled]
    assert extractor.dtypes == [torch.bfloat16, torch.bfloat16]


def test_feature_extraction_preserves_global_cpu_rng_state():
    torch.manual_seed(1234)
    state = torch.random.get_rng_state().clone()

    extract_location_features(
        _FeatureDataset(), _RngConsumingExtractor(), batch_size=2
    )

    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_feature_extraction_preserves_global_cuda_rng_state():
    torch.cuda.manual_seed_all(1234)
    states = [state.clone() for state in torch.cuda.get_rng_state_all()]

    extract_location_features(
        _FeatureDataset(),
        _RngConsumingExtractor(),
        batch_size=2,
        device='cuda',
    )

    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            torch.cuda.get_rng_state_all(), states, strict=True
        )
    )


def test_fixed_source_aligned_ridge_recovers_linear_targets():
    # 1. Give both sources shared tunnel information and source-specific agent data.
    rng = np.random.default_rng(7)
    shared = rng.normal(size=(96, 4))
    encoder = np.concatenate((shared, rng.normal(size=(96, 3))), axis=1)
    predictor = np.concatenate((shared, rng.normal(size=(96, 3))), axis=1)
    features = np.stack((encoder, predictor), axis=1)
    targets = np.empty((96, 8))
    targets[:, :2] = encoder @ rng.normal(size=(7, 2))
    targets[:, 2:4] = predictor @ rng.normal(size=(7, 2))
    targets[:, 4:] = shared @ rng.normal(size=(4, 4))
    targets += rng.normal(scale=1e-5, size=targets.shape)
    splits = np.repeat([0, 1, 2], [48, 16, 32])

    # 2. Fit one fixed-alpha ridge probe per visual source.
    models, result = fit_location_probe(
        features,
        targets,
        splits,
        TARGET_NAMES,
        ridge_alpha=1e-4,
    )

    # 3. Confirm split semantics and prevent each source from predicting wrong-time agents.
    assert result.ridge_alpha == 1e-4
    assert result.sample_counts == {'train': 64, 'test': 32}
    assert (
        'agent_future' not in result.test_metrics['current_visual']['points']
    )
    assert (
        'agent_current'
        not in result.test_metrics['predicted_visual']['points']
    )
    # 4. Each source should recover its aligned targets and shared tunnel targets.
    expected_outputs = {'current_visual': 6, 'predicted_visual': 6}
    for feature_index, source in enumerate(expected_outputs):
        metrics = result.test_metrics[source]
        assert metrics['average']['r2'] > 0.999
        assert metrics['average']['mae'] < 1e-3
        assert models[source].predict(features[:3, feature_index]).shape == (
            3,
            expected_outputs[source],
        )
    assert (
        result.test_metrics['current_visual']['points']['tunnel_green_hub'][
            'r2'
        ]
        > 0.999
    )
    assert (
        result.test_metrics['predicted_visual']['points']['tunnel_green_leaf'][
            'r2'
        ]
        > 0.999
    )
    # 5. Keep the W&B metric namespace explicit and free of retired aliases.
    flat = flatten_probe_metrics(result)
    assert 'location_probe_encoder/test/agent_current/mae' in flat
    assert 'location_probe_predictor/test/agent_future/mae' in flat
    assert 'location_probe_predictor/test/tunnel_green_hub/r2' in flat
    assert 'location_probe_encoder/test/average/mae' in flat
    assert not any('validation' in name for name in flat)
    assert not any('rmse' in name or 'baseline' in name for name in flat)
    assert not any('selected_alpha' in name for name in flat)
    assert not any(name.startswith('location_probe/') for name in flat)
    # 6. Keep provenance-style summary fields separate from evaluation metrics.
    summary = flatten_probe_summary(result)
    assert summary == {
        'location_probe/ridge_alpha': 1e-4,
        'location_probe/feature_width': 7,
        'location_probe/feature_readout': FEATURE_READOUT,
        'location_probe/samples/train': 64,
        'location_probe/samples/test': 32,
    }


def test_location_metrics_are_coordinate_mae_and_mean_xy_r2():
    metrics = regression_metrics(
        np.asarray([[0.0, 0.0], [2.0, 2.0]]),
        np.asarray([[1.0, 0.0], [0.0, 4.0]]),
        ('agent_current_x', 'agent_current_y'),
    )

    assert metrics['points']['agent_current']['mae'] == pytest.approx(1.25)
    assert metrics['points']['agent_current']['r2'] == pytest.approx(-1.25)
    assert metrics['average'] == metrics['points']['agent_current']


@pytest.mark.parametrize('ridge_alpha', [0.0, -1.0, np.nan, np.inf])
def test_location_probe_rejects_invalid_fixed_ridge_alpha(ridge_alpha):
    with pytest.raises(ValueError, match='finite and positive'):
        fit_location_probe(
            np.zeros((3, 2, 1)),
            np.zeros((3, len(TARGET_NAMES))),
            np.asarray([0, 1, 2]),
            TARGET_NAMES,
            ridge_alpha,
        )


def test_feature_cache_metadata_and_snapshot_are_immutable(tmp_path):
    # 1. Construct metadata that uniquely binds source, checkpoint, and samples.
    metadata = FeatureCacheMetadata(
        source_sha256='a' * 64,
        checkpoint_sha256='b' * 64,
        readout=FEATURE_READOUT,
        sample_sha256='c' * 64,
        feature_width=6,
    )
    path = snapshot_cache_path(
        tmp_path, metadata.checkpoint_sha256, batch_index=10_000
    )
    features = np.arange(48, dtype=np.float32).reshape(4, 2, 6)
    targets = np.arange(32, dtype=np.float32).reshape(4, 8)
    splits = np.asarray([0, 0, 1, 2], np.int8)
    sample_ids = np.arange(4, dtype=np.int64)

    # 2. Write once and verify an exact round trip under matching provenance.
    write_feature_cache(
        path, features, targets, splits, sample_ids, TARGET_NAMES, metadata
    )
    loaded = load_feature_cache(path, metadata)

    assert np.array_equal(loaded[0], features)
    assert loaded[-1] == TARGET_NAMES
    # 3. Metadata tampering must invalidate an otherwise readable cache.
    with h5py.File(path, 'r+') as cache:
        cache.attrs['schema_version'] = 1
    with pytest.raises(ValueError, match='metadata mismatch'):
        load_feature_cache(path, metadata)
    # 4. Neither snapshot allocation nor cache writing may overwrite evidence.
    with pytest.raises(FileExistsError):
        snapshot_cache_path(
            tmp_path, metadata.checkpoint_sha256, batch_index=10_000
        )
    with pytest.raises(FileExistsError):
        write_feature_cache(
            path, features, targets, splits, sample_ids, TARGET_NAMES, metadata
        )
