"""Immutable HDF5 and manifest artifacts for filled Tunnel Rooms markers."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from stable_worldmodel.envs.tunnel_rooms.rendering import (
    FILLED_AGENT_RENDER_RADIUS,
    filled_appearance,
    marker_appearance,
    render_filled_markers,
)


RERENDER_SCHEMA_VERSION = 2
SCENARIO_MANIFEST_SCHEMA_VERSION = 3
IMAGE_KEYS = ('pixels', 'goal')
DEFAULT_RENDERER_METHOD = 'exact_state_replay_and_visual_raster_upgrade'

RenderFrames = Callable[
    [np.ndarray, Mapping[str, Any], str, np.ndarray], np.ndarray
]


@dataclass(frozen=True)
class RerenderPaths:
    """All immutable and transactional paths for one derived dataset."""

    source: Path
    destination: Path
    journal: Path
    provenance: Path

    @property
    def partial(self) -> Path:
        return self.destination.with_suffix('.partial.h5')


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a new JSON file without replacing an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _lexists(path):
        raise FileExistsError(f'Refusing to replace existing JSON: {path}')
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp'
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w') as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _append_journal(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    journal_exists = _lexists(path)
    with path.open('a') as file:
        file.write(canonical_json(record) + '\n')
        file.flush()
        os.fsync(file.fileno())
    if not journal_exists:
        fsync_directory(path.parent)


def _load_journal(path: Path) -> list[dict[str, Any]]:
    """Load JSONL, repairing only an interrupted final record."""
    if not _lexists(path):
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    valid_bytes = 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError('Journal records must be JSON objects.')
            records.append(record)
            valid_bytes += len(line)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
            if index != len(lines) - 1:
                raise RuntimeError(
                    f'Corrupt rerender journal: {path}'
                ) from error
            with path.open('r+b') as file:
                file.truncate(valid_bytes)
                file.flush()
                os.fsync(file.fileno())
            break
    else:
        if lines and not data.endswith(b'\n'):
            with path.open('ab') as file:
                file.write(b'\n')
                file.flush()
                os.fsync(file.fileno())
    return records


def _dataset_value_hash(dataset: h5py.Dataset, chunk_rows: int = 4096) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset.dtype).encode('ascii'))
    digest.update(np.asarray(dataset.shape, np.int64).tobytes())
    for start in range(0, dataset.shape[0], chunk_rows):
        values = dataset[start : start + chunk_rows]
        if values.dtype.kind == 'O':
            for value in values.reshape(-1):
                encoded = (
                    value if isinstance(value, bytes) else str(value).encode()
                )
                digest.update(len(encoded).to_bytes(8, 'little'))
                digest.update(encoded)
        else:
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def validate_episode_structure(
    file: h5py.File,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate canonical offsets and row-aligned HDF5 columns."""
    required = {
        'ep_len',
        'ep_offset',
        'pos_agent',
        'pos_target',
        *IMAGE_KEYS,
    }
    missing = sorted(required - set(file.keys()))
    if missing:
        raise ValueError(f'HDF5 is missing required datasets: {missing}.')
    lengths = file['ep_len'][:].astype(np.int64)
    offsets = file['ep_offset'][:].astype(np.int64)
    expected = np.concatenate(([0], np.cumsum(lengths[:-1])))
    if not np.array_equal(offsets, expected):
        raise ValueError('HDF5 episode offsets are not canonical.')
    if len(lengths) == 0 or np.any(lengths <= 0):
        raise ValueError('HDF5 must contain non-empty episodes.')
    rows = int(lengths.sum())
    for name, dataset in file.items():
        expected_rows = (
            len(lengths) if name in {'ep_len', 'ep_offset'} else rows
        )
        if dataset.shape[0] != expected_rows:
            raise ValueError(
                f'HDF5 dataset {name!r} has {dataset.shape[0]} rows; '
                f'expected {expected_rows}.'
            )
    return lengths, offsets


def _scenario_fields(
    row: Mapping[str, Any], num_leaves: int
) -> dict[str, Any]:
    excluded = {
        'accepted_attempt',
        'accepted_noise_seed',
        'condition',
        'episode_sha256',
        'identity_mode',
        'retries',
        'source_episode',
        'steps',
        'success',
        'teleport_count',
        'timeout',
        'traces',
    }
    scenario = {
        key: value for key, value in row.items() if key not in excluded
    }
    scenario['num_leaves'] = num_leaves
    return scenario


def load_source_scenarios(
    manifest_path: Path, num_leaves: int, num_episodes: int
) -> tuple[list[dict[str, Any]], str]:
    """Load frozen scenarios from a scenario or paired audit manifest."""
    source_hash = sha256_file(manifest_path)
    document = json.loads(manifest_path.read_text())
    if 'scenarios' in document:
        scenarios = [
            _scenario_fields(row, num_leaves)
            for row in document['scenarios'][:num_episodes]
        ]
    elif 'episodes' in document:
        paired: dict[int, dict[str, dict[str, Any]]] = {}
        for row in document['episodes']:
            episode = int(row['episode'])
            if episode >= num_episodes:
                continue
            paired.setdefault(episode, {})[str(row['condition'])] = (
                _scenario_fields(row, num_leaves)
            )
        scenarios = []
        for episode in range(num_episodes):
            conditions = paired.get(episode, {})
            if set(conditions) != {'hidden', 'paired_color'}:
                raise ValueError(
                    f'Audit manifest lacks a paired scenario for episode {episode}.'
                )
            if conditions['hidden'] != conditions['paired_color']:
                raise ValueError(
                    f'Hidden/color scenario geometry differs at episode {episode}.'
                )
            scenarios.append(conditions['hidden'])
    else:
        raise ValueError(
            'Manifest contains neither scenarios nor audit episodes.'
        )
    if len(scenarios) != num_episodes:
        raise ValueError(
            f'Manifest supplies {len(scenarios)} scenarios; expected {num_episodes}.'
        )
    for episode, scenario in enumerate(scenarios):
        if int(scenario.get('episode', -1)) != episode:
            raise ValueError(
                'Scenario episodes must be contiguous and zero-based.'
            )
        endpoints = np.asarray(scenario.get('endpoint_positions'))
        if endpoints.shape != (2 * num_leaves, 2):
            raise ValueError(
                f'Invalid endpoint geometry at episode {episode}.'
            )
    return scenarios, source_hash


def upgrade_scenario_manifest(
    source_manifest: Path,
    destination: Path,
    num_leaves: int,
    num_episodes: int,
    *,
    appearance: Mapping[str, Any] | None = None,
    agent_render_radius: float = FILLED_AGENT_RENDER_RADIUS,
    sampling: str = 'exact_scenario_replay_from_outline_manifest',
    scenario_updates: Mapping[str, Any] | None = None,
    retain_legacy_marker_fields: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """Write a marker-upgraded scenario manifest at a deterministic path."""
    scenarios, source_hash = load_source_scenarios(
        source_manifest, num_leaves, num_episodes
    )
    marker_contract = dict(appearance or marker_appearance())
    updates = dict(scenario_updates or {})
    upgraded = []
    for scenario in scenarios:
        row = {**scenario, **marker_contract, **updates}
        row['agent_render_radius'] = agent_render_radius
        if retain_legacy_marker_fields:
            row['marker_size'] = 7.0
            row['marker_stroke_width'] = 1.0
        else:
            row.pop('marker_size', None)
            row.pop('marker_stroke_width', None)
        hash_input = {
            key: value
            for key, value in row.items()
            if key not in {'episode', 'scenario_hash'}
        }
        row['scenario_hash'] = sha256_json(hash_input)
        upgraded.append(row)
    stream = {
        'environment': 'swm/TunnelRooms-v1',
        'num_leaves': num_leaves,
        'marker_appearance': marker_contract,
        'agent_render_radius': agent_render_radius,
        'sampling': sampling,
        'source_manifest_sha256': source_hash,
    }
    if updates:
        stream['scenario_updates'] = updates
    output = {
        'schema_version': SCENARIO_MANIFEST_SCHEMA_VERSION,
        'num_scenarios': num_episodes,
        'marker_appearance': marker_contract,
        'agent_render_radius': agent_render_radius,
        'source_manifest': str(source_manifest.resolve()),
        'source_manifest_sha256': source_hash,
        'stream': stream,
        'stream_hash': sha256_json(stream),
        'scenarios': upgraded,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if json.loads(destination.read_text()) != output:
            raise FileExistsError(
                f'Existing upgraded manifest has different contents: {destination}'
            )
    else:
        try:
            atomic_write_json(destination, output)
        except FileExistsError:
            if json.loads(destination.read_text()) != output:
                raise
    return upgraded, source_hash


def _validate_path_layout(paths: RerenderPaths) -> None:
    if not paths.source.is_file():
        raise FileNotFoundError(paths.source)
    if paths.source.name.endswith('.partial.h5'):
        raise ValueError('A partial HDF5 cannot be used as a rerender source.')
    if paths.source.resolve() == paths.destination.resolve():
        raise ValueError('Source and destination paths must differ.')
    outputs = {
        'destination': paths.destination,
        'partial': paths.partial,
        'journal': paths.journal,
        'provenance': paths.provenance,
    }
    resolved_outputs = [path.resolve() for path in outputs.values()]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError('Transactional output paths must be distinct.')
    for name, candidate in outputs.items():
        if _lexists(candidate) and candidate.is_symlink():
            raise FileExistsError(
                f'Refusing symbolic-link {name} path: {candidate}'
            )
        if _lexists(candidate) and os.path.samefile(paths.source, candidate):
            if candidate == paths.destination:
                raise ValueError(
                    'Source and destination refer to the same inode.'
                )
            raise ValueError(
                f'Source aliases transactional output: {candidate}'
            )


def _create_partial(source: Path, partial: Path) -> None:
    partial.parent.mkdir(parents=True, exist_ok=True)
    if _lexists(partial):
        raise FileExistsError(f'Partial HDF5 already exists: {partial}')
    descriptor, temporary_name = tempfile.mkstemp(
        dir=partial.parent, prefix=f'.{partial.name}.', suffix='.tmp'
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with (
            h5py.File(source, 'r', swmr=True) as input_file,
            h5py.File(temporary, 'w', libver='latest') as output_file,
        ):
            for key, value in input_file.attrs.items():
                output_file.attrs[key] = value
            for name, dataset in input_file.items():
                if name not in IMAGE_KEYS:
                    input_file.copy(name, output_file, name=name)
                    continue
                chunks = dataset.chunks or (
                    min(100, dataset.shape[0]),
                    *dataset.shape[1:],
                )
                output = output_file.create_dataset(
                    name,
                    shape=dataset.shape,
                    maxshape=dataset.maxshape,
                    dtype=dataset.dtype,
                    chunks=chunks,
                    **hdf5plugin.Blosc(
                        cname='lz4',
                        clevel=5,
                        shuffle=hdf5plugin.Blosc.SHUFFLE,
                    ),
                )
                for key, value in dataset.attrs.items():
                    output.attrs[key] = value
            output_file.flush()
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, partial)
        fsync_directory(partial.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _journal_header(
    paths: RerenderPaths,
    source_sha256: str,
    source_manifest_sha256: str,
    render_manifest_sha256: str,
    condition: str,
    num_leaves: int,
    lengths: np.ndarray,
    renderer_appearance: Mapping[str, Any],
    renderer_method: str,
) -> dict[str, Any]:
    complete_appearance = dict(renderer_appearance)
    agent_render_radius = complete_appearance.pop(
        'agent_render_radius', FILLED_AGENT_RENDER_RADIUS
    )
    spec = {
        'schema_version': RERENDER_SCHEMA_VERSION,
        'source': str(paths.source.resolve()),
        'destination': str(paths.destination.resolve()),
        'source_sha256': source_sha256,
        'source_scenario_manifest_sha256': source_manifest_sha256,
        'render_scenario_manifest_sha256': render_manifest_sha256,
        'condition': condition,
        'num_leaves': num_leaves,
        'num_episodes': len(lengths),
        'num_rows': int(lengths.sum()),
        'marker_appearance': complete_appearance,
        'agent_render_radius': agent_render_radius,
        'image_keys': list(IMAGE_KEYS),
    }
    if renderer_method != DEFAULT_RENDERER_METHOD:
        spec['renderer_method'] = renderer_method
    return {'type': 'header', 'spec': spec, 'spec_sha256': sha256_json(spec)}


@dataclass(frozen=True)
class _JournalState:
    next_episode: int
    validated: dict[str, Any] | None
    published: bool


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _journal_state(
    records: Sequence[Mapping[str, Any]],
    lengths: np.ndarray,
    destination: Path,
) -> _JournalState:
    next_episode = 0
    validated: dict[str, Any] | None = None
    published = False
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))
    for record in records[1:]:
        record_type = record.get('type')
        if record_type == 'episode':
            if (
                validated is not None
                or published
                or next_episode >= len(lengths)
            ):
                raise RuntimeError(
                    'Episode commit appears after finalization.'
                )
            expected = {
                'episode': next_episode,
                'row_start': int(offsets[next_episode]),
                'row_stop': int(offsets[next_episode] + lengths[next_episode]),
            }
            if any(
                int(record.get(key, -1)) != value
                for key, value in expected.items()
            ):
                raise RuntimeError(
                    'Non-contiguous rerender commit at episode '
                    f'{next_episode}.'
                )
            image_hashes = record.get('image_sha256')
            if not isinstance(image_hashes, dict) or set(image_hashes) != set(
                IMAGE_KEYS
            ):
                raise RuntimeError('Episode commit has invalid image hashes.')
            if not all(_is_sha256(value) for value in image_hashes.values()):
                raise RuntimeError('Episode commit has invalid image hashes.')
            next_episode += 1
            continue
        if record_type == 'validated':
            if next_episode != len(lengths) or published:
                raise RuntimeError(
                    'Validation record appears before all episode commits.'
                )
            if not _is_sha256(
                record.get('destination_sha256')
            ) or not _is_sha256(record.get('validation_sha256')):
                raise RuntimeError('Validation record has invalid hashes.')
            candidate = dict(record)
            if validated is not None and candidate != validated:
                raise RuntimeError('Conflicting rerender validation records.')
            validated = candidate
            continue
        if record_type == 'published':
            if validated is None or published:
                raise RuntimeError(
                    'Invalid rerender publication record order.'
                )
            if record.get('destination') != str(destination.resolve()):
                raise RuntimeError(
                    'Rerender publication record has the wrong destination.'
                )
            published = True
            continue
        raise RuntimeError(
            f'Unknown rerender journal record: {record_type!r}.'
        )
    return _JournalState(
        next_episode=next_episode,
        validated=validated,
        published=published,
    )


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode):
        raise FileExistsError(f'Expected a regular artifact file: {path}')
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
    )


def _verified_file_hash(path: Path) -> str:
    before = _file_identity(path)
    digest = sha256_file(path)
    if _file_identity(path) != before:
        raise RuntimeError(f'Artifact changed while it was verified: {path}')
    return digest


def verify_file_hash(path: Path, expected: str, label: str) -> None:
    actual = _verified_file_hash(path)
    if actual != expected:
        raise FileExistsError(
            f'{label} collision has hash {actual}; expected {expected}: {path}'
        )


def _verify_matching_json(
    path: Path, expected: Mapping[str, Any], label: str
) -> None:
    try:
        actual = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FileExistsError(
            f'Existing {label} is not the expected JSON artifact: {path}'
        ) from error
    if actual != expected:
        raise FileExistsError(
            f'Existing {label} has different contents: {path}'
        )


def publish_json_idempotently(
    path: Path, value: Mapping[str, Any], label: str = 'JSON'
) -> None:
    if _lexists(path):
        _verify_matching_json(path, value, label)
        fsync_directory(path.parent)
        return
    try:
        atomic_write_json(path, value)
    except FileExistsError:
        _verify_matching_json(path, value, label)
        fsync_directory(path.parent)


def _render_episode(
    source: h5py.File,
    destination: h5py.File,
    row_start: int,
    row_stop: int,
    scenario: Mapping[str, Any],
    condition: str,
    chunk_rows: int,
    render_frames: RenderFrames,
) -> dict[str, str]:
    hashes = {}
    for key in IMAGE_KEYS:
        digest = hashlib.sha256()
        for start in range(row_start, row_stop, chunk_rows):
            stop = min(start + chunk_rows, row_stop)
            positions = (
                source['pos_agent'][start:stop]
                if key == 'pixels'
                else source['pos_target'][start:stop]
            )
            rendered = render_frames(
                source[key][start:stop], scenario, condition, positions
            )
            destination[key][start:stop] = rendered
            digest.update(np.ascontiguousarray(rendered).tobytes())
        hashes[key] = digest.hexdigest()
    return hashes


def validate_rerender(
    source_path: Path,
    derived_path: Path,
    scenarios: Sequence[Mapping[str, Any]],
    condition: str,
    chunk_rows: int = 32,
    *,
    render_frames: RenderFrames = render_filled_markers,
    renderer_appearance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fully validate schema, non-image values, and every rendered frame."""
    with (
        h5py.File(source_path, 'r', swmr=True) as source,
        h5py.File(derived_path, 'r', swmr=True) as derived,
    ):
        lengths, offsets = validate_episode_structure(source)
        other_lengths, other_offsets = validate_episode_structure(derived)
        if set(source.keys()) != set(derived.keys()):
            raise ValueError('Source and derived HDF5 keys differ.')
        if not np.array_equal(lengths, other_lengths) or not np.array_equal(
            offsets, other_offsets
        ):
            raise ValueError('Episode structure changed during rerendering.')
        nonimage_hashes = {}
        for key in source:
            if source[key].shape != derived[key].shape:
                raise ValueError(f'Dataset shape changed for {key}.')
            if source[key].dtype != derived[key].dtype:
                raise ValueError(f'Dataset dtype changed for {key}.')
            if key in IMAGE_KEYS:
                continue
            source_hash = _dataset_value_hash(source[key])
            if source_hash != _dataset_value_hash(derived[key]):
                raise ValueError(
                    f'Non-image values changed for dataset {key}.'
                )
            nonimage_hashes[key] = source_hash
        image_delta = {}
        for key in IMAGE_KEYS:
            changed_values = 0
            changed_frames = 0
            for episode, (offset, length) in enumerate(
                zip(offsets, lengths, strict=True)
            ):
                row_stop = int(offset + length)
                for start in range(int(offset), row_stop, chunk_rows):
                    stop = min(start + chunk_rows, row_stop)
                    original = source[key][start:stop]
                    positions = (
                        source['pos_agent'][start:stop]
                        if key == 'pixels'
                        else source['pos_target'][start:stop]
                    )
                    expected = render_frames(
                        original, scenarios[episode], condition, positions
                    )
                    actual = derived[key][start:stop]
                    if not np.array_equal(actual, expected):
                        raise ValueError(
                            f'{key} differs from exact rendering at rows '
                            f'{start}:{stop}.'
                        )
                    delta = actual != original
                    changed_values += int(np.count_nonzero(delta))
                    changed_frames += int(
                        np.count_nonzero(np.any(delta, axis=(1, 2, 3)))
                    )
            if changed_values == 0 or changed_frames == 0:
                raise ValueError(
                    f'Rerendering did not change any {key} pixels.'
                )
            image_delta[key] = {
                'changed_values': changed_values,
                'changed_frames': changed_frames,
                'total_frames': int(lengths.sum()),
            }
    return {
        'num_episodes': len(lengths),
        'num_rows': int(lengths.sum()),
        'nonimage_value_sha256': nonimage_hashes,
        'image_delta': image_delta,
        'appearance': dict(renderer_appearance or filled_appearance()),
    }


def _source_identity(path: Path) -> tuple[int, int, int, int]:
    details = path.stat()
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
    )


def _require_unchanged_source(
    path: Path,
    expected_hash: str,
    expected_identity: tuple[int, int, int, int],
) -> None:
    if (
        sha256_file(path) != expected_hash
        or _source_identity(path) != expected_identity
    ):
        raise RuntimeError('Source HDF5 changed during rerendering.')


def rerender_dataset(
    paths: RerenderPaths,
    scenarios: Sequence[Mapping[str, Any]],
    source_manifest_sha256: str,
    render_manifest_sha256: str,
    condition: str,
    num_leaves: int,
    *,
    chunk_rows: int = 32,
    after_episode: Callable[[int], None] | None = None,
    render_frames: RenderFrames = render_filled_markers,
    renderer_appearance: Mapping[str, Any] | None = None,
    renderer_method: str = DEFAULT_RENDERER_METHOD,
) -> dict[str, Any]:
    """Resume, verify, and publish one immutable image-only derived HDF5."""
    _validate_path_layout(paths)
    source_identity = _source_identity(paths.source)
    source_hash = sha256_file(paths.source)
    with h5py.File(paths.source, 'r', swmr=True) as source:
        lengths, offsets = validate_episode_structure(source)
    if len(scenarios) != len(lengths):
        raise ValueError('Scenario count does not match source episodes.')
    appearance = dict(renderer_appearance or filled_appearance())
    header = _journal_header(
        paths,
        source_hash,
        source_manifest_sha256,
        render_manifest_sha256,
        condition,
        num_leaves,
        lengths,
        appearance,
        renderer_method,
    )
    records = _load_journal(paths.journal)
    if records:
        if records[0] != header:
            raise RuntimeError(
                'Rerender journal does not match this derivation.'
            )
    else:
        if _lexists(paths.partial):
            raise RuntimeError(
                'Partial HDF5 and rerender journal must either both exist or '
                f'both be absent: {paths.partial}, {paths.journal}'
            )
        if _lexists(paths.destination):
            raise FileExistsError(
                f'Destination already exists: {paths.destination}'
            )
        if _lexists(paths.provenance):
            raise FileExistsError(
                f'Provenance already exists: {paths.provenance}'
            )
        _append_journal(paths.journal, header)
        records = [header]
    state = _journal_state(records, lengths, paths.destination)
    if _lexists(paths.provenance) and not state.published:
        raise FileExistsError(
            'Provenance exists before the journal records publication: '
            f'{paths.provenance}'
        )
    if state.validated is None:
        if _lexists(paths.destination):
            raise FileExistsError(
                f'Destination already exists: {paths.destination}'
            )
        if not _lexists(paths.partial):
            if state.next_episode != 0:
                raise RuntimeError(
                    'Committed rerender episodes are missing their partial HDF5.'
                )
            _create_partial(paths.source, paths.partial)
        with (
            h5py.File(paths.source, 'r', swmr=True) as source,
            h5py.File(paths.partial, 'r+') as derived,
        ):
            partial_lengths, partial_offsets = validate_episode_structure(
                derived
            )
            if not np.array_equal(
                lengths, partial_lengths
            ) or not np.array_equal(offsets, partial_offsets):
                raise RuntimeError(
                    'Partial HDF5 episode structure is invalid.'
                )
            for episode in range(state.next_episode, len(lengths)):
                row_start = int(offsets[episode])
                row_stop = row_start + int(lengths[episode])
                hashes = _render_episode(
                    source,
                    derived,
                    row_start,
                    row_stop,
                    scenarios[episode],
                    condition,
                    chunk_rows,
                    render_frames,
                )
                derived.flush()
                descriptor = os.open(paths.partial, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                _append_journal(
                    paths.journal,
                    {
                        'type': 'episode',
                        'episode': episode,
                        'row_start': row_start,
                        'row_stop': row_stop,
                        'image_sha256': hashes,
                    },
                )
                if after_episode is not None:
                    after_episode(episode)
        validation_path = paths.partial
    else:
        if state.published and not _lexists(paths.destination):
            raise RuntimeError(
                'Journal records publication but destination is missing: '
                f'{paths.destination}'
            )
        if not _lexists(paths.partial) and not _lexists(paths.destination):
            raise RuntimeError(
                'Validated rerender has neither a partial nor destination HDF5.'
            )
        validation_path = (
            paths.partial if _lexists(paths.partial) else paths.destination
        )
        verify_file_hash(
            validation_path,
            state.validated['destination_sha256'],
            'Validated rerender artifact',
        )
    validation = validate_rerender(
        paths.source,
        validation_path,
        scenarios,
        condition,
        chunk_rows,
        render_frames=render_frames,
        renderer_appearance=appearance,
    )
    _require_unchanged_source(paths.source, source_hash, source_identity)
    validation_hash = sha256_json(validation)
    if state.validated is None:
        destination_hash = _verified_file_hash(paths.partial)
        validated_record = {
            'type': 'validated',
            'destination_sha256': destination_hash,
            'validation_sha256': validation_hash,
        }
        _append_journal(paths.journal, validated_record)
        state = _JournalState(
            next_episode=len(lengths),
            validated=validated_record,
            published=False,
        )
    else:
        destination_hash = state.validated['destination_sha256']
        if validation_hash != state.validated['validation_sha256']:
            raise RuntimeError(
                'Recovered rerender validation does not match its journal.'
            )
    if _lexists(paths.destination):
        verify_file_hash(paths.destination, destination_hash, 'Destination')
        if _lexists(paths.partial) and not os.path.samefile(
            paths.partial, paths.destination
        ):
            raise FileExistsError(
                'Destination collision is not the published partial inode: '
                f'{paths.destination}'
            )
    else:
        if state.published:
            raise RuntimeError(
                'Journal records publication but destination is missing: '
                f'{paths.destination}'
            )
        if not _lexists(paths.partial):
            raise RuntimeError('Validated partial HDF5 is missing.')
        verify_file_hash(paths.partial, destination_hash, 'Partial HDF5')
        try:
            os.link(paths.partial, paths.destination)
        except FileExistsError:
            verify_file_hash(
                paths.destination, destination_hash, 'Destination'
            )
            if not os.path.samefile(paths.partial, paths.destination):
                raise FileExistsError(
                    'Destination collision is not the published partial inode: '
                    f'{paths.destination}'
                )
    fsync_directory(paths.destination.parent)
    if _lexists(paths.partial):
        if not os.path.samefile(paths.partial, paths.destination):
            raise RuntimeError(
                'Partial HDF5 no longer aliases the published destination.'
            )
        paths.partial.unlink()
        fsync_directory(paths.partial.parent)
    if not state.published:
        _append_journal(
            paths.journal,
            {
                'type': 'published',
                'destination': str(paths.destination.resolve()),
            },
        )
    _require_unchanged_source(paths.source, source_hash, source_identity)
    provenance = {
        'schema_version': RERENDER_SCHEMA_VERSION,
        'source_hdf5': str(paths.source.resolve()),
        'source_sha256': source_hash,
        'destination_hdf5': str(paths.destination.resolve()),
        'destination_sha256': destination_hash,
        'source_scenario_manifest_sha256': source_manifest_sha256,
        'render_scenario_manifest_sha256': render_manifest_sha256,
        'condition': condition,
        'num_leaves': num_leaves,
        'renderer': {
            **appearance,
            'method': renderer_method,
        },
        'validation': validation,
    }
    publish_json_idempotently(paths.provenance, provenance, 'provenance')
    return provenance


__all__ = [
    'DEFAULT_RENDERER_METHOD',
    'RERENDER_SCHEMA_VERSION',
    'RenderFrames',
    'RerenderPaths',
    'atomic_write_json',
    'fsync_directory',
    'load_source_scenarios',
    'publish_json_idempotently',
    'rerender_dataset',
    'sha256_file',
    'upgrade_scenario_manifest',
    'validate_episode_structure',
    'validate_rerender',
    'verify_file_hash',
]
