from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts.data import filled_marker_artifacts as artifact_module
from scripts.data import rerender_tunnel_rooms as rerender_module
from scripts.data.filled_marker_artifacts import (
    RerenderPaths,
    load_source_scenarios,
    rerender_dataset,
    sha256_file,
    upgrade_scenario_manifest,
)
from scripts.data.filled_marker_sidecars import (
    SidecarSettings,
    build_paired_location_sidecars,
    semantic_endpoint_order,
)
from scripts.data.rerender_tunnel_rooms import (
    FILLED_PROFILE,
    finalize_provenance,
    run_from_config,
)
from stable_worldmodel.envs.tunnel_rooms.rendering import (
    AGENT_COLOR,
    FILLED_AGENT_RENDER_RADIUS,
    endpoint_colors,
    marker_bounds,
    render_filled_markers,
)


def _scenarios(count: int, num_leaves: int = 2) -> list[dict]:
    endpoints = (
        [[14.2, 14.4], [14.1, 37.3], [37.4, 14.2], [37.1, 37.4]]
        if num_leaves == 2
        else [
            [14.2, 12.4],
            [14.1, 26.3],
            [14.3, 40.2],
            [37.4, 12.2],
            [37.1, 26.4],
            [37.2, 40.1],
        ]
    )
    return [
        {
            'episode': episode,
            'num_leaves': num_leaves,
            'scenario_seed': 1000 + episode,
            'pairing': episode % (2 if num_leaves == 2 else 6),
            'endpoint_positions': endpoints,
            'agent_position': [24.0, 24.0],
            'target_position': [42.0, 42.0],
            'target_leaf': 1,
            'landing_offset': 14.0,
            'trigger_radius': 10.0,
            'rearm_distance': 28.0,
            'min_hub_separation': 20.0,
            'speed': 5.0,
        }
        for episode in range(count)
    ]


def _outline_frame(
    scenario: dict, condition: str, position: np.ndarray
) -> np.ndarray:
    image = np.full((56, 56, 3), [244, 245, 241], dtype=np.uint8)
    colors = endpoint_colors(
        condition, scenario['num_leaves'], scenario['pairing']
    )
    for endpoint, color in zip(
        scenario['endpoint_positions'], colors, strict=True
    ):
        center_x = int(np.floor(endpoint[0] + 0.5))
        center_y = int(np.floor(endpoint[1] + 0.5))
        image[center_y - 7 : center_y + 8, center_x - 7 : center_x + 8] = color
        image[center_y - 6 : center_y + 7, center_x - 6 : center_x + 7] = [
            244,
            245,
            241,
        ]
    x, y = np.floor(position + 0.5).astype(int)
    image[y, x] = AGENT_COLOR
    return image


def test_semantic_sidecar_labels_follow_explicit_color_names():
    scenario = {
        **_scenarios(1)[0],
        'tunnel_color_names': ['green', 'blue'],
    }

    labels = semantic_endpoint_order(scenario, 'paired_color')

    assert [name for name, _hub, _leaf in labels] == ['green', 'blue']


def test_recorded_hash_verification_preserves_finalizer_contract(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / 'artifact.json'
    artifact.write_text('original')
    alias = tmp_path / 'artifact-link.json'
    alias.symlink_to(artifact)

    rerender_module._require_recorded_hash(alias, sha256_file(artifact))

    with pytest.raises(RuntimeError, match='Provenance hash mismatch'):
        rerender_module._require_recorded_hash(artifact, '0' * 64)
    with pytest.raises(
        FileNotFoundError, match='Provenance artifact is missing'
    ):
        rerender_module._require_recorded_hash(
            tmp_path / 'missing.json', '0' * 64
        )


def _write_source(
    path: Path,
    scenarios: list[dict],
    condition: str = 'hidden',
    episode_length: int = 8,
) -> None:
    lengths = np.full(len(scenarios), episode_length, dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int64)
    positions = []
    pixels = []
    goals = []
    for scenario in scenarios:
        goal_position = np.asarray(scenario['target_position'], np.float32)
        goal = _outline_frame(scenario, condition, goal_position)
        for step in range(episode_length):
            position = np.asarray([24.0 + step / 10, 24.0], np.float32)
            positions.append(position)
            pixels.append(_outline_frame(scenario, condition, position))
            goals.append(goal)
    positions_array = np.asarray(positions, dtype=np.float32)
    total = len(positions_array)
    episode_ids = np.repeat(np.arange(len(scenarios), dtype=np.int32), lengths)
    with h5py.File(path, 'x') as file:
        file.attrs['fixture'] = 'outline'
        file.create_dataset('ep_len', data=lengths, maxshape=(None,))
        file.create_dataset('ep_offset', data=offsets, maxshape=(None,))
        file.create_dataset(
            'pixels',
            data=np.asarray(pixels),
            maxshape=(None, 56, 56, 3),
            chunks=(2, 56, 56, 3),
        )
        file.create_dataset(
            'goal',
            data=np.asarray(goals),
            maxshape=(None, 56, 56, 3),
            chunks=(2, 56, 56, 3),
        )
        file.create_dataset(
            'pos_agent', data=positions_array, maxshape=(None, 2)
        )
        file.create_dataset(
            'pos_target',
            data=np.full((total, 2), [42.0, 42.0], np.float32),
            maxshape=(None, 2),
        )
        file.create_dataset(
            'proprio', data=positions_array, maxshape=(None, 2)
        )
        file.create_dataset(
            'action', data=np.zeros((total, 2), np.float32), maxshape=(None, 2)
        )
        file.create_dataset(
            'reward', data=np.arange(total, dtype=np.float64), maxshape=(None,)
        )
        file.create_dataset(
            'terminated', data=np.zeros(total, dtype=bool), maxshape=(None,)
        )
        file.create_dataset(
            'truncated', data=np.zeros(total, dtype=bool), maxshape=(None,)
        )
        file.create_dataset('ep_idx', data=episode_ids, maxshape=(None,))
        file.create_dataset(
            'step_idx',
            data=np.tile(np.arange(episode_length), len(scenarios)),
            maxshape=(None,),
        )
        file.create_dataset(
            'id', data=np.arange(total, dtype=np.int64), maxshape=(None,)
        )
        file.create_dataset(
            'env_name',
            data=np.asarray(['TunnelRooms'] * total, dtype=object),
            dtype=h5py.string_dtype(),
            maxshape=(None,),
        )


def _paths(tmp_path: Path, source: Path) -> RerenderPaths:
    return RerenderPaths(
        source=source,
        destination=tmp_path / 'datasets' / 'derived.h5',
        journal=tmp_path / 'run' / 'rerender.jsonl',
        provenance=tmp_path / 'run' / 'provenance.json',
    )


def test_journal_repairs_unterminated_valid_record_before_append(tmp_path):
    journal = tmp_path / 'rerender.jsonl'
    records = [{'type': 'header'}, {'type': 'episode', 'episode': 0}]
    journal.write_text('\n'.join(map(artifact_module.canonical_json, records)))

    assert artifact_module._load_journal(journal) == records
    artifact_module._append_journal(journal, {'type': 'validated'})

    assert journal.read_bytes().endswith(b'\n')
    assert [
        json.loads(line) for line in journal.read_bytes().splitlines()
    ] == [
        *records,
        {'type': 'validated'},
    ]


def test_filled_renderer_is_exact_18x18_with_radius9_agent():
    scenario = _scenarios(1)[0]
    source = _outline_frame(scenario, 'hidden', np.asarray([14.0, 14.0]))
    position = np.asarray([14.0, 14.0], np.float32)
    rendered = render_filled_markers(source, scenario, 'hidden', position)
    rows, columns = marker_bounds(
        scenario['endpoint_positions'][0], source.shape[:2]
    )

    assert rendered[rows, columns].shape == (18, 18, 3)
    assert np.array_equal(rendered[14, 14], AGENT_COLOR)
    grid_y, grid_x = np.ogrid[: source.shape[0], : source.shape[1]]
    agent = (grid_x - position[0]) ** 2 + (
        grid_y - position[1]
    ) ** 2 <= FILLED_AGENT_RENDER_RADIUS**2
    assert np.all(rendered[agent] == AGENT_COLOR)
    marker_agent = agent[rows, columns]
    assert np.all(rendered[rows, columns][~marker_agent] == [62, 94, 112])
    assert np.array_equal(rendered[0, 0], source[0, 0])


@pytest.mark.parametrize('condition', ['hidden', 'paired_color'])
@pytest.mark.parametrize('pairing', [0, 5])
def test_offline_renderer_matches_environment_filled_contract(
    condition, pairing
):
    from stable_worldmodel.envs.tunnel_rooms import TunnelRoomsEnv

    outline = TunnelRoomsEnv(identity_mode=condition, num_leaves=3)
    filled = TunnelRoomsEnv(identity_mode=condition, num_leaves=3)
    try:
        outline.reset(seed=41021)
        scenario = outline.scenario_spec()
        filled_scenario = {
            **scenario,
            'pairing': pairing,
            'marker_style': 'filled18x18',
            'marker_width': 18,
            'marker_height': 18,
            'marker_size': 7.0,
            'agent_render_radius': FILLED_AGENT_RENDER_RADIUS,
        }
        filled.reset(seed=41021, options={'scenario': filled_scenario})

        actual = render_filled_markers(
            outline.render(),
            filled_scenario,
            condition,
            outline.agent_position,
        )

        assert np.array_equal(actual, filled.render())
    finally:
        outline.close()
        filled.close()


def test_filled_renderer_changes_only_markers_and_agent():
    scenario = _scenarios(1)[0]
    source = np.random.default_rng(418).integers(
        0, 256, size=(2, 64, 64, 3), dtype=np.uint8
    )
    positions = np.asarray([[24.0, 24.0], [32.0, 32.0]], np.float32)
    rendered = render_filled_markers(
        source, scenario, 'paired_color', positions
    )

    dirty = np.zeros(source.shape[:3], dtype=bool)
    for endpoint in scenario['endpoint_positions']:
        rows, columns = marker_bounds(endpoint, source.shape[1:3])
        dirty[:, rows, columns] = True
    grid_y, grid_x = np.ogrid[: source.shape[1], : source.shape[2]]
    for frame, (center_x, center_y) in enumerate(positions):
        dirty[frame] |= (grid_x - center_x) ** 2 + (
            grid_y - center_y
        ) ** 2 <= FILLED_AGENT_RENDER_RADIUS**2

    np.testing.assert_array_equal(rendered[~dirty], source[~dirty])


def test_rerender_recovers_after_interruption_and_preserves_source(tmp_path):
    scenarios = _scenarios(5)
    source = tmp_path / 'outline.h5'
    _write_source(source, scenarios)
    paths = _paths(tmp_path, source)
    source_hash = sha256_file(source)
    source_stat = source.stat()

    def interrupt(episode: int) -> None:
        if episode == 1:
            raise RuntimeError('simulated interruption')

    with pytest.raises(RuntimeError, match='simulated interruption'):
        rerender_dataset(
            paths,
            scenarios,
            'a' * 64,
            'b' * 64,
            'hidden',
            2,
            chunk_rows=2,
            after_episode=interrupt,
        )

    assert paths.partial.exists()
    assert paths.journal.exists()
    assert not paths.destination.exists()
    assert sha256_file(source) == source_hash
    with paths.journal.open('ab') as file:
        file.write(b'{"type":"interrupted"')
        file.flush()
        os.fsync(file.fileno())

    provenance = rerender_dataset(
        paths,
        scenarios,
        'a' * 64,
        'b' * 64,
        'hidden',
        2,
        chunk_rows=2,
    )

    assert paths.destination.exists()
    assert paths.provenance.exists()
    assert not paths.partial.exists()
    assert (
        provenance['validation']['image_delta']['pixels']['changed_frames']
        == 40
    )
    assert sha256_file(source) == source_hash
    assert source.stat().st_ino == source_stat.st_ino
    assert source.stat().st_mtime_ns == source_stat.st_mtime_ns
    with (
        h5py.File(source, 'r') as outline,
        h5py.File(paths.destination, 'r') as filled,
    ):
        for key in outline:
            if key not in {'pixels', 'goal'}:
                assert np.array_equal(outline[key][:], filled[key][:])


def _inject_transaction_failure(
    monkeypatch: pytest.MonkeyPatch, stage: str, paths: RerenderPaths
) -> None:
    fired = False

    def fail_once() -> None:
        nonlocal fired
        if not fired:
            fired = True
            raise RuntimeError(f'simulated {stage} interruption')

    if stage in {'after_header', 'after_partial'}:
        original = artifact_module._create_partial

        def create_partial(source: Path, partial: Path) -> None:
            if stage == 'after_header':
                fail_once()
            original(source, partial)
            if stage == 'after_partial':
                fail_once()

        monkeypatch.setattr(artifact_module, '_create_partial', create_partial)
        return
    if stage in {'after_validated', 'after_published'}:
        original = artifact_module._append_journal
        record_type = stage.removeprefix('after_')

        def append_journal(path: Path, record: dict) -> None:
            original(path, record)
            if record.get('type') == record_type:
                fail_once()

        monkeypatch.setattr(artifact_module, '_append_journal', append_journal)
        return
    if stage == 'after_destination_link':
        original = artifact_module.os.link

        def link(source: Path, destination: Path) -> None:
            original(source, destination)
            if Path(destination) == paths.destination:
                fail_once()

        monkeypatch.setattr(artifact_module.os, 'link', link)
        return
    if stage == 'after_destination_fsync':
        original = artifact_module.fsync_directory

        def fsync_directory(path: Path) -> None:
            original(path)
            if paths.destination.exists() and paths.partial.exists():
                fail_once()

        monkeypatch.setattr(
            artifact_module, 'fsync_directory', fsync_directory
        )
        return
    if stage == 'after_partial_unlink':
        original = Path.unlink

        def unlink(path: Path, *args, **kwargs) -> None:
            original(path, *args, **kwargs)
            if path == paths.partial:
                fail_once()

        monkeypatch.setattr(Path, 'unlink', unlink)
        return
    if stage in {'before_provenance', 'after_provenance'}:
        original = artifact_module.atomic_write_json

        def atomic_write_json(path: Path, value: dict) -> None:
            if path == paths.provenance and stage == 'before_provenance':
                fail_once()
            original(path, value)
            if path == paths.provenance and stage == 'after_provenance':
                fail_once()

        monkeypatch.setattr(
            artifact_module, 'atomic_write_json', atomic_write_json
        )
        return
    raise AssertionError(f'Unknown failure stage: {stage}')


@pytest.mark.parametrize(
    'stage',
    [
        'after_header',
        'after_partial',
        'after_validated',
        'after_destination_link',
        'after_destination_fsync',
        'after_partial_unlink',
        'after_published',
        'before_provenance',
        'after_provenance',
    ],
)
def test_rerender_recovers_every_transaction_phase_without_replacing_final(
    tmp_path, monkeypatch, stage
):
    scenarios = _scenarios(1)
    source = tmp_path / 'outline.h5'
    _write_source(source, scenarios, episode_length=2)
    paths = _paths(tmp_path, source)
    source_identity = (source.stat().st_ino, source.stat().st_mtime_ns)
    source_hash = sha256_file(source)
    _inject_transaction_failure(monkeypatch, stage, paths)

    with pytest.raises(RuntimeError, match=f'simulated {stage} interruption'):
        rerender_dataset(
            paths, scenarios, 'a' * 64, 'b' * 64, 'hidden', 2, chunk_rows=1
        )

    interrupted_destination_inode = (
        paths.destination.stat().st_ino if paths.destination.exists() else None
    )
    provenance = rerender_dataset(
        paths, scenarios, 'a' * 64, 'b' * 64, 'hidden', 2, chunk_rows=1
    )
    published_inode = paths.destination.stat().st_ino
    provenance_bytes = paths.provenance.read_bytes()

    assert not paths.partial.exists()
    assert provenance['destination_sha256'] == sha256_file(paths.destination)
    assert (source.stat().st_ino, source.stat().st_mtime_ns) == source_identity
    assert sha256_file(source) == source_hash
    if interrupted_destination_inode is not None:
        assert published_inode == interrupted_destination_inode

    repeated = rerender_dataset(
        paths, scenarios, 'a' * 64, 'b' * 64, 'hidden', 2, chunk_rows=1
    )

    assert repeated == provenance
    assert paths.destination.stat().st_ino == published_inode
    assert paths.provenance.read_bytes() == provenance_bytes
    records = [
        json.loads(line) for line in paths.journal.read_text().splitlines()
    ]
    assert [record['type'] for record in records].count('validated') == 1
    assert [record['type'] for record in records].count('published') == 1


def test_rerender_refuses_unrelated_destination_during_recovery(
    tmp_path, monkeypatch
):
    scenarios = _scenarios(1)
    source = tmp_path / 'outline.h5'
    _write_source(source, scenarios, episode_length=2)
    paths = _paths(tmp_path, source)
    _inject_transaction_failure(monkeypatch, 'after_validated', paths)

    with pytest.raises(RuntimeError, match='after_validated interruption'):
        rerender_dataset(
            paths, scenarios, 'a' * 64, 'b' * 64, 'hidden', 2, chunk_rows=1
        )
    paths.destination.write_bytes(b'unrelated destination')
    collision_inode = paths.destination.stat().st_ino
    collision_hash = sha256_file(paths.destination)

    with pytest.raises(FileExistsError, match='Destination collision'):
        rerender_dataset(
            paths, scenarios, 'a' * 64, 'b' * 64, 'hidden', 2, chunk_rows=1
        )

    assert paths.destination.stat().st_ino == collision_inode
    assert sha256_file(paths.destination) == collision_hash
    assert paths.partial.exists()
    assert not paths.provenance.exists()


def test_rerender_refuses_unrelated_provenance_without_touching_final(
    tmp_path, monkeypatch
):
    scenarios = _scenarios(1)
    source = tmp_path / 'outline.h5'
    _write_source(source, scenarios, episode_length=2)
    paths = _paths(tmp_path, source)
    _inject_transaction_failure(monkeypatch, 'before_provenance', paths)

    with pytest.raises(RuntimeError, match='before_provenance interruption'):
        rerender_dataset(
            paths, scenarios, 'a' * 64, 'b' * 64, 'hidden', 2, chunk_rows=1
        )
    destination_inode = paths.destination.stat().st_ino
    destination_hash = sha256_file(paths.destination)
    paths.provenance.write_text('{"unrelated": true}\n')

    with pytest.raises(FileExistsError, match='different contents'):
        rerender_dataset(
            paths, scenarios, 'a' * 64, 'b' * 64, 'hidden', 2, chunk_rows=1
        )

    assert paths.destination.stat().st_ino == destination_inode
    assert sha256_file(paths.destination) == destination_hash
    assert json.loads(paths.provenance.read_text()) == {'unrelated': True}


def test_rerender_rejects_partial_source_collisions_and_same_inode(tmp_path):
    scenarios = _scenarios(1)
    source = tmp_path / 'source.h5'
    _write_source(source, scenarios)

    same_path = _paths(tmp_path, source)
    same_path = RerenderPaths(
        source=source,
        destination=source,
        journal=same_path.journal,
        provenance=same_path.provenance,
    )
    with pytest.raises(ValueError, match='paths must differ'):
        rerender_dataset(same_path, scenarios, 'a' * 64, 'b' * 64, 'hidden', 2)

    hardlink = tmp_path / 'same-inode.h5'
    os.link(source, hardlink)
    collision = RerenderPaths(
        source=source,
        destination=hardlink,
        journal=tmp_path / 'inode.jsonl',
        provenance=tmp_path / 'inode.json',
    )
    with pytest.raises(ValueError, match='same inode'):
        rerender_dataset(collision, scenarios, 'a' * 64, 'b' * 64, 'hidden', 2)

    partial_source = tmp_path / 'source.partial.h5'
    _write_source(partial_source, scenarios)
    partial_paths = _paths(tmp_path / 'partial-case', partial_source)
    with pytest.raises(ValueError, match='partial HDF5'):
        rerender_dataset(
            partial_paths, scenarios, 'a' * 64, 'b' * 64, 'hidden', 2
        )

    orphan_paths = _paths(tmp_path / 'orphan-case', source)
    orphan_paths.partial.parent.mkdir(parents=True)
    orphan_paths.partial.touch()
    with pytest.raises(RuntimeError, match='must either both exist'):
        rerender_dataset(
            orphan_paths, scenarios, 'a' * 64, 'b' * 64, 'hidden', 2
        )


def test_manifest_upgrade_supports_paired_audit_and_is_idempotent(tmp_path):
    scenarios = _scenarios(3)
    episodes = []
    for scenario in scenarios:
        for condition in ('hidden', 'paired_color'):
            episodes.append(
                {
                    **scenario,
                    'condition': condition,
                    'identity_mode': condition,
                    'success': True,
                    'steps': 8,
                    'source_episode': scenario['episode'],
                }
            )
    source_manifest = tmp_path / 'audit_manifest.json'
    source_manifest.write_text(json.dumps({'episodes': episodes}))
    destination = tmp_path / 'run' / 'scenario_manifest.json'

    loaded, source_hash = load_source_scenarios(source_manifest, 2, 3)
    upgraded, repeated_hash = upgrade_scenario_manifest(
        source_manifest, destination, 2, 3
    )
    repeated, _ = upgrade_scenario_manifest(source_manifest, destination, 2, 3)

    assert loaded[0]['num_leaves'] == 2
    assert source_hash == repeated_hash
    assert upgraded == repeated
    manifest = json.loads(destination.read_text())
    assert manifest['source_manifest_sha256'] == source_hash
    assert manifest['scenarios'][0]['marker_style'] == 'filled18x18'
    assert manifest['scenarios'][0]['marker_width'] == 18
    assert manifest['scenarios'][0]['agent_render_radius'] == 9
    assert len(manifest['scenarios'][0]['scenario_hash']) == 64


def test_paired_sidecars_share_exact_samples_splits_rows_and_targets(tmp_path):
    scenarios = _scenarios(5)
    outline = tmp_path / 'outline.h5'
    _write_source(outline, scenarios, condition='paired_color')
    paths = _paths(tmp_path, outline)
    provenance = rerender_dataset(
        paths,
        scenarios,
        'a' * 64,
        'b' * 64,
        'paired_color',
        2,
        chunk_rows=2,
    )
    outline_sidecar = tmp_path / 'run' / 'outline_location.h5'
    filled_sidecar = tmp_path / 'run' / 'filled_location.h5'
    settings = SidecarSettings(
        fit_episodes=2,
        validation_episodes=1,
        test_episodes=2,
        history_size=3,
        frameskip=2,
        num_preds=1,
        seed=77,
    )

    result = build_paired_location_sidecars(
        outline,
        paths.destination,
        outline_sidecar,
        filled_sidecar,
        scenarios,
        'paired_color',
        settings,
        outline_sha256=provenance['source_sha256'],
        filled_sha256=provenance['destination_sha256'],
        include_positional_targets=True,
    )

    assert result['sample_counts'] == {'fit': 2, 'validation': 1, 'test': 2}
    assert 'tunnel_green_hub_x' in result['target_names']
    assert 'tunnel_orange_leaf_y' in result['target_names']
    assert 'tunnel_top_hub_x' in result['target_names']
    assert 'tunnel_bottom_leaf_y' in result['target_names']
    with (
        h5py.File(outline_sidecar, 'r') as old,
        h5py.File(filled_sidecar, 'r') as new,
    ):
        assert old.attrs['schema_version'] == 2
        assert new.attrs['schema_version'] == 2
        for key in (
            'sample_id',
            'episode_id',
            'split',
            'frame_rows',
            'future_row',
            'target',
        ):
            assert old[key][:].tobytes() == new[key][:].tobytes()
        assert old.attrs['sample_sha256'] == new.attrs['sample_sha256']
        assert old.attrs['source_hdf5'] != new.attrs['source_hdf5']
        assert old.attrs['source_sha256'] != new.attrs['source_sha256']
        assert np.bincount(old['split'][:], minlength=3).tolist() == [2, 1, 2]
        assert np.all(np.diff(old['frame_rows'][:], axis=1) == 2)
        assert np.array_equal(
            old['future_row'][:], old['frame_rows'][:, -1] + 2
        )
        names = json.loads(old.attrs['target_names_json'])
        targets = old['target'][:]
        np.testing.assert_allclose(
            targets[:, names.index('tunnel_top_leaf_y')], 14.2
        )
        np.testing.assert_allclose(
            targets[:, names.index('tunnel_bottom_leaf_y')], 37.4
        )
        swapped = old['episode_id'][:] % 2 == 1
        assert np.all(
            targets[swapped, names.index('tunnel_green_leaf_y')]
            == targets[swapped, names.index('tunnel_bottom_leaf_y')]
        )


def test_config_has_canonical_two_leaf_output_and_deterministic_root():
    config_path = (
        Path(__file__).parents[2]
        / 'scripts/data/config/tunnel_rooms_filled18x18.yaml'
    )
    cfg = OmegaConf.load(config_path)
    datasets = [
        dataset
        for group in cfg.leaf_groups.values()
        for dataset in group.datasets.values()
    ]
    assert [dataset.output_name for dataset in datasets] == [
        'tunnel_room_color_filled18x18.h5'
    ]
    assert cfg.profile == FILLED_PROFILE
    assert cfg.groups == ['two_leaf']
    assert cfg.conditions == ['paired_color']
    assert cfg.agent_render_radius == 9
    assert [dataset.include_positional_targets for dataset in datasets] == [
        True
    ]
    assert [dataset.destination_sidecar_name for dataset in datasets] == [
        'location_probe.h5'
    ]


def test_palette_render_respects_pairing_and_preserves_input():
    scenario = _scenarios(2)[1]
    palette = [[40, 153, 78], [83, 98, 216]]
    scenario['tunnel_colors'] = palette
    source = np.full((1, 56, 56, 3), 240, dtype=np.uint8)
    original = source.copy()
    rendered = render_filled_markers(
        source, scenario, 'paired_color', np.array([[26.0, 26.0]])
    )
    for endpoint, color in zip(
        scenario['endpoint_positions'],
        [palette[0], palette[1], palette[1], palette[0]],
        strict=True,
    ):
        x, y = np.floor(np.array(endpoint) + 0.5).astype(int)
        np.testing.assert_array_equal(rendered[0, y, x], color)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize('palette_override', [False, True])
def test_run_writes_no_clobber_aggregate_provenance(
    tmp_path, palette_override
):
    scenarios = _scenarios(5)
    datasets = tmp_path / 'datasets'
    datasets.mkdir()
    source = datasets / 'outline.h5'
    _write_source(source, scenarios, condition='paired_color')
    source_manifest = tmp_path / 'source_scenarios.json'
    source_manifest.write_text(json.dumps({'scenarios': scenarios}))
    cfg = OmegaConf.create(
        {
            'output_root': str(tmp_path),
            'profile': FILLED_PROFILE,
            'groups': ['two_leaf'],
            'conditions': ['paired_color'],
            'chunk_rows': 2,
            'finalize_only': False,
            'agent_render_radius': 9,
            'location_probe': {
                'fit_episodes': 2,
                'validation_episodes': 1,
                'test_episodes': 2,
                'history_size': 3,
                'frameskip': 2,
                'num_preds': 1,
                'seed': 55,
            },
            'leaf_groups': {
                'two_leaf': {
                    'num_leaves': 2,
                    'source_manifest': str(source_manifest),
                    'datasets': {
                        'paired_color': {
                            'source': str(source),
                            'output_name': 'filled.h5',
                            'source_sidecar_name': 'outline_sidecar.h5',
                            'destination_sidecar_name': 'filled_sidecar.h5',
                            'include_positional_targets': True,
                        }
                    },
                }
            },
        }
    )

    if palette_override:
        cfg.palette = {
            'colors': [[40, 153, 78], [83, 98, 216]],
            'names': ['green', 'blue'],
        }
    runtime_name = (
        'tunnel_rooms_palette'
        if palette_override
        else 'tunnel_rooms_filled18x18'
    )
    result = run_from_config(cfg)

    condition_manifest = (
        tmp_path
        / f'runs/{runtime_name}/two_leaf/paired_color/provenance_manifest.json'
    )
    assert condition_manifest.is_file()
    aggregate_path = finalize_provenance(cfg)
    aggregate = json.loads(aggregate_path.read_text())
    dataset = aggregate['groups']['two_leaf']['datasets']['paired_color']
    assert aggregate['renderer']['marker_style'] == 'filled18x18'
    assert aggregate['renderer']['agent_render_radius'] == 9
    assert dataset['provenance']['validation']['image_delta']['pixels']
    assert len(dataset['location_sidecars']['outline_sidecar_sha256']) == 64
    assert len(dataset['location_sidecars']['filled_sidecar_sha256']) == 64
    assert dataset['provenance']['source_scenario_manifest_sha256']
    assert dataset['provenance']['render_scenario_manifest_sha256']
    assert result['two_leaf']['datasets']['paired_color']
    filled_sidecar = Path(dataset['location_sidecars']['filled_sidecar'])
    with h5py.File(filled_sidecar, 'r') as sidecar:
        target_names = json.loads(sidecar.attrs['target_names_json'])
        assert sidecar.attrs['schema_version'] == 2
        assert 'tunnel_green_hub_x' in target_names
        assert 'tunnel_top_hub_x' in target_names
        if palette_override:
            assert 'tunnel_blue_leaf_x' in target_names
            assert 'tunnel_orange_leaf_x' not in target_names
    if palette_override:
        assert aggregate['renderer']['palette']['colors'] == cfg.palette.colors
        with (
            h5py.File(source, 'r') as old,
            h5py.File(tmp_path / 'datasets/filled.h5', 'r') as new,
        ):
            np.testing.assert_array_equal(old['action'][:], new['action'][:])
        manifest = json.loads(
            (
                tmp_path
                / 'runs'
                / runtime_name
                / 'two_leaf/scenario_manifest.json'
            ).read_text()
        )
        assert manifest['scenarios'][0]['tunnel_color_names'] == [
            'green',
            'blue',
        ]

    condition_bytes = condition_manifest.read_bytes()
    aggregate_bytes = aggregate_path.read_bytes()
    recovered = run_from_config(cfg)
    recovered_aggregate = finalize_provenance(cfg)

    assert recovered == result
    assert condition_manifest.read_bytes() == condition_bytes
    assert recovered_aggregate.read_bytes() == aggregate_bytes
