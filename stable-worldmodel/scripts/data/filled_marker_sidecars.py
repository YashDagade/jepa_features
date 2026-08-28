"""Deterministic location-label sidecars for outline/filled controls."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from scripts.data.filled_marker_artifacts import (
    fsync_directory,
    load_source_scenarios,
    sha256_file,
    validate_episode_structure,
)
from stable_worldmodel.envs.tunnel_rooms.rendering import (
    PAIRED_SHAPE_ORDER,
    SEMANTIC_COLOR_NAMES,
    shape_identity_indices,
)


SIDECAR_SCHEMA_VERSION = 1
CUED_SIDECAR_SCHEMA_VERSION = 2
SPLIT_NAMES = ('fit', 'validation', 'test')

SemanticEndpointOrder = Callable[
    [Mapping[str, object], str],
    list[tuple[str, np.ndarray, np.ndarray]],
]


def paired_shape_endpoint_order(
    scenario: Mapping[str, object], condition: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return paired shape hub and leaf coordinates for probe labels."""
    if condition != 'paired_shape':
        raise ValueError(f'Unknown Tunnel Rooms condition: {condition}')
    endpoints, num_leaves = _scenario_endpoints(scenario)
    identities = shape_identity_indices(num_leaves, int(scenario['pairing']))
    hubs = endpoints[:num_leaves]
    leaves = endpoints[num_leaves:]
    leaf_identities = identities[num_leaves:]
    return [
        (
            PAIRED_SHAPE_ORDER[identity],
            hubs[identity],
            leaves[leaf_identities.index(identity)],
        )
        for identity in range(num_leaves)
    ]


def semantic_endpoint_order(
    scenario: Mapping[str, object], condition: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return deterministic semantic hub and leaf pairs for probe labels."""
    endpoints, num_leaves = _scenario_endpoints(scenario)
    hubs = endpoints[:num_leaves]
    leaves = endpoints[num_leaves:]
    if condition == 'paired_color':
        names = tuple(
            str(name)
            for name in scenario.get(
                'tunnel_color_names', SEMANTIC_COLOR_NAMES[:num_leaves]
            )
        )
        if len(names) != num_leaves or len(set(names)) != num_leaves:
            raise ValueError(
                'tunnel_color_names must provide one unique name per leaf.'
            )
        permutation = tuple(itertools.permutations(range(num_leaves)))[
            int(scenario['pairing'])
        ]
        cue_assignment = tuple(
            int(value)
            for value in scenario.get('cue_assignment', range(num_leaves))
        )
        if sorted(cue_assignment) != list(range(num_leaves)):
            raise ValueError(
                'cue_assignment must be a complete cue permutation.'
            )
        hub_for_semantic = {
            semantic_index: hub_index
            for hub_index, semantic_index in enumerate(cue_assignment)
        }
        return [
            (
                name,
                hubs[hub_for_semantic[semantic_index]],
                leaves[permutation[hub_for_semantic[semantic_index]]],
            )
            for semantic_index, name in enumerate(names)
        ]
    if condition != 'hidden':
        raise ValueError(f'Unknown Tunnel Rooms condition: {condition}')
    return positional_endpoint_order(scenario)


def positional_endpoint_order(
    scenario: Mapping[str, object],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Name hub and leaf endpoints independently from top to bottom."""
    endpoints, num_leaves = _scenario_endpoints(scenario)
    names_by_leaves = {
        1: ('single',),
        2: ('top', 'bottom'),
        3: ('top', 'middle', 'bottom'),
        4: ('top', 'upper_middle', 'lower_middle', 'bottom'),
    }
    try:
        names = names_by_leaves[num_leaves]
    except KeyError as error:
        raise ValueError(
            f'Unsupported Tunnel Rooms leaf count: {num_leaves}'
        ) from error
    hubs = endpoints[:num_leaves]
    leaves = endpoints[num_leaves:]
    return [
        (name, hubs[hub_index], leaves[leaf_index])
        for name, hub_index, leaf_index in zip(
            names,
            np.argsort(hubs[:, 1], kind='stable'),
            np.argsort(leaves[:, 1], kind='stable'),
            strict=True,
        )
    ]


@dataclass(frozen=True)
class SidecarSettings:
    """Fixed sampling contract shared by outline and filled controls."""

    fit_episodes: int = 512
    validation_episodes: int = 128
    test_episodes: int = 256
    history_size: int = 3
    frameskip: int = 5
    num_preds: int = 1
    seed: int = 180018

    @property
    def total_episodes(self) -> int:
        return (
            self.fit_episodes + self.validation_episodes + self.test_episodes
        )

    @property
    def final_offset(self) -> int:
        return (self.history_size - 1 + self.num_preds) * self.frameskip


@dataclass(frozen=True)
class _SidecarTable:
    """Deterministic sample rows and labels for one location sidecar."""

    sample_ids: np.ndarray
    episode_ids: np.ndarray
    split_codes: np.ndarray
    frame_rows: np.ndarray
    future_rows: np.ndarray
    targets: np.ndarray
    target_names: tuple[str, ...]
    sample_sha256: str
    schema_version: int


def _stable_clip_start(seed: int, episode: int, valid_starts: int) -> int:
    payload = f'{seed}:{episode}'.encode('ascii')
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
    return value % valid_starts


def _scenario_endpoints(
    scenario: Mapping[str, object],
) -> tuple[np.ndarray, int]:
    """Read the common endpoint layout used by all sidecar label orders."""
    endpoints = np.asarray(scenario['endpoint_positions'], dtype=np.float32)
    num_leaves = int(scenario.get('num_leaves', len(endpoints) // 2))
    if endpoints.shape != (2 * num_leaves, 2):
        raise ValueError(
            f'Expected {(2 * num_leaves, 2)} endpoints, got {endpoints.shape}.'
        )
    return endpoints, num_leaves


def _read_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 rows while satisfying h5py's sorted-index rule."""
    requested = np.asarray(rows, dtype=np.int64)
    unique, inverse = np.unique(requested, return_inverse=True)
    return dataset[unique][inverse].reshape(
        (*requested.shape, *dataset.shape[1:])
    )


def _location_targets(
    scenarios: Sequence[Mapping[str, object]],
    condition: str,
    source: h5py.File,
    episode_ids: np.ndarray,
    frame_rows: np.ndarray,
    future_rows: np.ndarray,
    semantic_order: SemanticEndpointOrder,
    include_positional_targets: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    names = [
        'agent_current_x',
        'agent_current_y',
        'agent_future_x',
        'agent_future_y',
    ]

    def endpoint_order(
        scenario: Mapping[str, object],
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        order = semantic_order(scenario, condition)
        if include_positional_targets:
            if condition == 'hidden':
                raise ValueError(
                    'Hidden sidecars already use positional tunnel labels.'
                )
            order = [*order, *positional_endpoint_order(scenario)]
        names = [item[0] for item in order]
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate tunnel label names: {names}.')
        return order

    first = endpoint_order(scenarios[int(episode_ids[0])])
    for name, _hub, _leaf in first:
        names.extend(
            (
                f'tunnel_{name}_hub_x',
                f'tunnel_{name}_hub_y',
                f'tunnel_{name}_leaf_x',
                f'tunnel_{name}_leaf_y',
            )
        )
    targets = np.empty((len(episode_ids), len(names)), dtype=np.float32)
    expected_order = [item[0] for item in first]
    for index, episode in enumerate(episode_ids):
        current = source['pos_agent'][frame_rows[index, -1]]
        future = source['pos_agent'][future_rows[index]]
        values = [*current.tolist(), *future.tolist()]
        order = endpoint_order(scenarios[int(episode)])
        if [item[0] for item in order] != expected_order:
            raise RuntimeError(
                'Location label ordering changed between scenarios.'
            )
        for _name, hub, leaf in order:
            values.extend((*hub.tolist(), *leaf.tolist()))
        targets[index] = values
    return targets, tuple(names)


def _sample_sha256(
    sample_ids: np.ndarray,
    episode_ids: np.ndarray,
    frame_rows: np.ndarray,
    future_rows: np.ndarray,
    splits: np.ndarray,
    targets: np.ndarray,
    target_names: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ('sample_id', sample_ids),
        ('episode_id', episode_ids),
        ('frame_rows', frame_rows),
        ('future_row', future_rows),
        ('split', splits),
        ('target', targets),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode('ascii'))
        digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(
        json.dumps(list(target_names), separators=(',', ':')).encode()
    )
    return digest.hexdigest()


def _write_sidecar(
    path: Path,
    source_hdf5: Path,
    source_sha256: str,
    sample_ids: np.ndarray,
    episode_ids: np.ndarray,
    split_codes: np.ndarray,
    frame_rows: np.ndarray,
    future_rows: np.ndarray,
    targets: np.ndarray,
    target_names: Sequence[str],
    settings: SidecarSettings,
    sample_hash: str,
    schema_version: int,
) -> None:
    partial = path.with_suffix('.partial.h5')
    path.parent.mkdir(parents=True, exist_ok=True)

    expected_attrs = {
        'schema_version': schema_version,
        'source_hdf5': str(source_hdf5.resolve()),
        'source_sha256': source_sha256,
        'sample_sha256': sample_hash,
        'frameskip': settings.frameskip,
        'num_preds': settings.num_preds,
        'history_size': settings.history_size,
        'target_names_json': json.dumps(list(target_names)),
    }
    expected_datasets = {
        'sample_id': sample_ids,
        'episode_id': episode_ids,
        'split': split_codes,
        'frame_rows': frame_rows,
        'future_row': future_rows,
        'target': targets,
    }

    def verify_existing(candidate: Path) -> None:
        try:
            details = candidate.lstat()
            if not stat.S_ISREG(details.st_mode):
                raise ValueError('sidecar path is not a regular file')
            before = (
                details.st_dev,
                details.st_ino,
                details.st_size,
                details.st_mtime_ns,
            )
            with h5py.File(candidate, 'r') as file:
                if set(file.attrs) != set(expected_attrs):
                    raise ValueError('sidecar attributes differ')
                for name, expected in expected_attrs.items():
                    if file.attrs[name] != expected:
                        raise ValueError(f'sidecar attribute {name!r} differs')
                if set(file.keys()) != set(expected_datasets):
                    raise ValueError('sidecar datasets differ')
                for name, expected in expected_datasets.items():
                    dataset = file[name]
                    if (
                        dataset.shape != expected.shape
                        or dataset.dtype != expected.dtype
                        or not np.array_equal(dataset[:], expected)
                    ):
                        raise ValueError(f'sidecar dataset {name!r} differs')
            after_details = candidate.lstat()
            after = (
                after_details.st_dev,
                after_details.st_ino,
                after_details.st_size,
                after_details.st_mtime_ns,
            )
            if after != before:
                raise RuntimeError('sidecar changed while it was verified')
        except FileExistsError:
            raise
        except (
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            raise FileExistsError(
                'Existing location sidecar is not the expected artifact: '
                f'{candidate}'
            ) from error

    partial_exists = os.path.lexists(partial)
    final_exists = os.path.lexists(path)
    if partial_exists:
        verify_existing(partial)
    if final_exists:
        verify_existing(path)
    if partial_exists and final_exists and not os.path.samefile(partial, path):
        raise FileExistsError(
            'Location sidecar final is not the published partial inode: '
            f'{path}'
        )
    if final_exists and not partial_exists:
        with path.open('rb') as file:
            os.fsync(file.fileno())
        fsync_directory(path.parent)
        return

    if not partial_exists:
        with h5py.File(partial, 'x') as file:
            for name, value in expected_attrs.items():
                file.attrs[name] = value
            for name, value in expected_datasets.items():
                file.create_dataset(name, data=value)
            file.flush()
        verify_existing(partial)

    with partial.open('rb') as file:
        os.fsync(file.fileno())
    fsync_directory(partial.parent)
    if not final_exists:
        try:
            os.link(partial, path)
        except FileExistsError:
            verify_existing(path)
            if not os.path.samefile(partial, path):
                raise FileExistsError(
                    'Location sidecar final is not the published partial inode: '
                    f'{path}'
                )
    fsync_directory(path.parent)
    if os.path.lexists(partial):
        if not os.path.samefile(partial, path):
            raise RuntimeError(
                'Location sidecar partial no longer aliases its final path.'
            )
        partial.unlink()
        fsync_directory(partial.parent)
    fsync_directory(path.parent)


def _build_sidecar_table(
    source: h5py.File,
    scenarios: Sequence[Mapping[str, object]],
    condition: str,
    settings: SidecarSettings,
    semantic_order: SemanticEndpointOrder,
    include_positional_targets: bool,
) -> _SidecarTable:
    """Select one deterministic clip per episode and construct its labels."""
    lengths, offsets = validate_episode_structure(source)
    if len(scenarios) != len(lengths):
        raise ValueError('Scenario count does not match sidecar source.')
    eligible = np.flatnonzero(lengths > settings.final_offset)
    if len(eligible) < settings.total_episodes:
        raise ValueError(
            f'Only {len(eligible)} episodes support a four-frame clip; '
            f'{settings.total_episodes} required.'
        )

    rng = np.random.default_rng(settings.seed)
    episode_ids = rng.permutation(eligible)[: settings.total_episodes].astype(
        np.int64
    )
    split_codes = np.concatenate(
        (
            np.full(settings.fit_episodes, 0, np.int8),
            np.full(settings.validation_episodes, 1, np.int8),
            np.full(settings.test_episodes, 2, np.int8),
        )
    )
    frame_rows = np.empty(
        (settings.total_episodes, settings.history_size), np.int64
    )
    future_rows = np.empty(settings.total_episodes, np.int64)
    for index, episode in enumerate(episode_ids):
        valid_starts = int(lengths[episode]) - settings.final_offset
        local_start = _stable_clip_start(
            settings.seed, int(episode), valid_starts
        )
        absolute_start = int(offsets[episode]) + local_start
        frame_rows[index] = (
            absolute_start
            + np.arange(settings.history_size, dtype=np.int64)
            * settings.frameskip
        )
        future_rows[index] = (
            frame_rows[index, -1] + settings.num_preds * settings.frameskip
        )

    targets, target_names = _location_targets(
        scenarios,
        condition,
        source,
        episode_ids,
        frame_rows,
        future_rows,
        semantic_order,
        include_positional_targets,
    )
    sample_ids = np.arange(settings.total_episodes, dtype=np.int64)
    sample_hash = _sample_sha256(
        sample_ids,
        episode_ids,
        frame_rows,
        future_rows,
        split_codes,
        targets,
        target_names,
    )
    return _SidecarTable(
        sample_ids=sample_ids,
        episode_ids=episode_ids,
        split_codes=split_codes,
        frame_rows=frame_rows,
        future_rows=future_rows,
        targets=targets,
        target_names=target_names,
        sample_sha256=sample_hash,
        schema_version=(
            CUED_SIDECAR_SCHEMA_VERSION
            if include_positional_targets
            else SIDECAR_SCHEMA_VERSION
        ),
    )


def _write_sidecar_table(
    path: Path,
    source_hdf5: Path,
    source_sha256: str,
    table: _SidecarTable,
    settings: SidecarSettings,
) -> None:
    _write_sidecar(
        path,
        source_hdf5,
        source_sha256,
        table.sample_ids,
        table.episode_ids,
        table.split_codes,
        table.frame_rows,
        table.future_rows,
        table.targets,
        table.target_names,
        settings,
        table.sample_sha256,
        table.schema_version,
    )


def _manifest_num_leaves(path: Path) -> int:
    """Read the tunnel count needed by the shared manifest validator."""
    document = json.loads(path.read_text())
    if 'num_leaves' in document:
        return int(document['num_leaves'])
    rows = document.get('scenarios') or document.get('episodes')
    if not rows:
        raise ValueError(f'Scenario manifest contains no scenarios: {path}')
    first = rows[0]
    if 'num_leaves' in first:
        return int(first['num_leaves'])
    endpoints = np.asarray(first.get('endpoint_positions'))
    if endpoints.ndim != 2 or endpoints.shape[0] % 2:
        raise ValueError(f'Cannot infer tunnel count from manifest: {path}')
    return endpoints.shape[0] // 2


def build_location_sidecar(
    source_hdf5: Path,
    scenario_manifest: Path,
    sidecar: Path,
    condition: str,
    settings: SidecarSettings,
    *,
    source_sha256: str | None = None,
) -> dict[str, object]:
    """Build one deterministic probe sidecar from an HDF5 and its manifest.

    Hidden datasets receive top-to-bottom positional tunnel labels. Paired
    color datasets receive both color-identity and positional tunnel labels.
    """
    if condition not in {'hidden', 'paired_color'}:
        raise ValueError(
            'Single-source sidecars support only hidden or paired_color; '
            f'got {condition!r}.'
        )
    source_hash = source_sha256 or sha256_file(source_hdf5)
    with h5py.File(source_hdf5, 'r', swmr=True) as source:
        lengths, _offsets = validate_episode_structure(source)
        scenarios, _manifest_hash = load_source_scenarios(
            scenario_manifest,
            _manifest_num_leaves(scenario_manifest),
            len(lengths),
        )
        table = _build_sidecar_table(
            source,
            scenarios,
            condition,
            settings,
            semantic_endpoint_order,
            include_positional_targets=condition == 'paired_color',
        )
    _write_sidecar_table(sidecar, source_hdf5, source_hash, table, settings)
    return {
        'sidecar': str(sidecar.resolve()),
        'sidecar_sha256': sha256_file(sidecar),
        'source_sha256': source_hash,
        'sample_sha256': table.sample_sha256,
        'sample_counts': {
            name: count
            for name, count in zip(
                SPLIT_NAMES,
                (
                    settings.fit_episodes,
                    settings.validation_episodes,
                    settings.test_episodes,
                ),
                strict=True,
            )
        },
        'target_names': list(table.target_names),
    }


def build_paired_location_sidecars(
    outline_hdf5: Path,
    filled_hdf5: Path,
    outline_sidecar: Path,
    filled_sidecar: Path,
    scenarios: Sequence[Mapping[str, object]],
    condition: str,
    settings: SidecarSettings,
    *,
    outline_sha256: str | None = None,
    filled_sha256: str | None = None,
    semantic_order: SemanticEndpointOrder = semantic_endpoint_order,
    include_positional_targets: bool = False,
) -> dict[str, object]:
    """Write outline/filled sidecars with byte-identical sample tables."""
    if outline_hdf5.resolve() == filled_hdf5.resolve():
        raise ValueError('Outline and filled sidecar sources must differ.')
    outline_hash = outline_sha256 or sha256_file(outline_hdf5)
    filled_hash = filled_sha256 or sha256_file(filled_hdf5)
    with (
        h5py.File(outline_hdf5, 'r', swmr=True) as outline,
        h5py.File(filled_hdf5, 'r', swmr=True) as filled,
    ):
        lengths, offsets = validate_episode_structure(outline)
        other_lengths, other_offsets = validate_episode_structure(filled)
        if not np.array_equal(lengths, other_lengths) or not np.array_equal(
            offsets, other_offsets
        ):
            raise ValueError('Outline and filled episode structures differ.')
        if len(scenarios) != len(lengths):
            raise ValueError('Scenario count does not match sidecar sources.')
        table = _build_sidecar_table(
            outline,
            scenarios,
            condition,
            settings,
            semantic_order,
            include_positional_targets,
        )
        if not np.array_equal(
            _read_rows(outline['pos_agent'], table.frame_rows),
            _read_rows(filled['pos_agent'], table.frame_rows),
        ):
            raise ValueError('Filled HDF5 changed location-probe labels.')
    _write_sidecar_table(
        outline_sidecar, outline_hdf5, outline_hash, table, settings
    )
    _write_sidecar_table(
        filled_sidecar, filled_hdf5, filled_hash, table, settings
    )
    with (
        h5py.File(outline_sidecar, 'r') as outline,
        h5py.File(filled_sidecar, 'r') as filled,
    ):
        for key in (
            'sample_id',
            'episode_id',
            'split',
            'frame_rows',
            'future_row',
            'target',
        ):
            if not np.array_equal(outline[key][:], filled[key][:]):
                raise RuntimeError(f'Paired sidecars differ in {key}.')
    return {
        'outline_sidecar': str(outline_sidecar.resolve()),
        'filled_sidecar': str(filled_sidecar.resolve()),
        'outline_sidecar_sha256': sha256_file(outline_sidecar),
        'filled_sidecar_sha256': sha256_file(filled_sidecar),
        'outline_source_sha256': outline_hash,
        'filled_source_sha256': filled_hash,
        'sample_sha256': table.sample_sha256,
        'sample_counts': {
            name: count
            for name, count in zip(
                SPLIT_NAMES,
                (
                    settings.fit_episodes,
                    settings.validation_episodes,
                    settings.test_episodes,
                ),
                strict=True,
            )
        },
        'target_names': list(table.target_names),
    }
