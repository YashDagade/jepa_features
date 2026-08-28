"""Persisted-data contracts and storage for paired Tunnel Rooms collection.

Constants and canonical JSON helpers define the manifest contract. The
collection journal is the transaction boundary: an episode pair is durable
only after both HDF5 appends succeed and the matching ``pair_commit`` record is
fsynced. On restart, HDF5 files are truncated back to that committed boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf

from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from stable_worldmodel.envs.tunnel_rooms import marker_raster


# -----------------------------------------------------------------------------
# Stable persisted-data contract
# -----------------------------------------------------------------------------


# Version written into current scenario, evaluation, and audit manifests.
SCHEMA_VERSION = 2

# Deterministic training scenarios, frozen before any rollout is collected.
SCENARIO_MANIFEST_NAME = 'scenario_manifest.json'

# Final collection checks and per-episode rollout audit records.
AUDIT_MANIFEST_NAME = 'audit_manifest.json'

# Deterministic rank-stratified evaluation scenarios.
EVAL_MANIFEST_NAME = 'eval_scenarios.json'

# Append-only transaction log used to resume paired collection safely.
JOURNAL_NAME = 'collection_journal.jsonl'

# Matched render conditions written together by every pair transaction.
CONDITIONS = ('hidden', 'paired_color')
SUPPORTED_CONDITIONS = frozenset(CONDITIONS)

# Learner inputs that every stored episode must contain.
TRAINING_KEYS = ('pixels', 'action', 'proprio')


def conditions_from_cfg(cfg: DictConfig) -> tuple[str, ...]:
    """Return the explicit collection conditions, preserving paired defaults."""
    configured = cfg.get('conditions')
    conditions = (
        CONDITIONS if configured is None else tuple(map(str, configured))
    )
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError('Collection conditions must be non-empty and unique.')
    unsupported = sorted(set(conditions) - SUPPORTED_CONDITIONS)
    if unsupported:
        raise ValueError(f'Unsupported collection conditions: {unsupported}.')
    return conditions


def _json_default(value: Any) -> Any:
    """Convert supported scientific Python values to ordinary JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__} to JSON.')


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashing and exact equality checks."""
    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(',', ':'),
    )


def atomic_write_json(path: Path, value: Any) -> None:
    """Write indented JSON, then atomically publish it at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    with temporary.open('w') as file:
        json.dump(value, file, default=_json_default, indent=2)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


# -----------------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetFiles:
    """Published and in-progress paths for one collection condition."""

    final: Path
    partial: Path

    def existing(self) -> Path | None:
        """Return the sole existing path, rejecting an ambiguous state."""
        if self.final.exists() and self.partial.exists():
            raise RuntimeError(
                f'Both final and partial dataset files exist: {self.final}'
            )
        if self.partial.exists():
            return self.partial
        return self.final if self.final.exists() else None


DatasetPaths = dict[str, DatasetFiles]


def policy_settings(cfg: DictConfig | None = None) -> dict[str, float]:
    """Normalize policy noise settings used by rollouts and journal identity."""
    policy = cfg.get('policy') if cfg is not None else {}
    policy = policy or {}
    return {
        'action_noise': float(policy.get('action_noise', 0.0)),
        'action_repeat_prob': float(policy.get('action_repeat_prob', 0.0)),
    }


def dataset_paths(cfg: DictConfig) -> DatasetPaths:
    """Resolve final and transactional HDF5 paths for every condition."""
    root = Path(cfg.dataset_cache_dir) / 'datasets'
    root.mkdir(parents=True, exist_ok=True)
    return {
        condition: DatasetFiles(
            final=root / f'{cfg.dataset_names[condition]}.h5',
            partial=root / f'{cfg.dataset_names[condition]}.partial.h5',
        )
        for condition in conditions_from_cfg(cfg)
    }


# -----------------------------------------------------------------------------
# Journal records
# -----------------------------------------------------------------------------


def _collection_spec(cfg: DictConfig) -> dict[str, Any]:
    """Return every setting that must match when a collection resumes."""
    policy = policy_settings(cfg)
    conditions = conditions_from_cfg(cfg)
    prefix_cfg = cfg.get('prefix')
    prefix = None
    if prefix_cfg is not None and bool(prefix_cfg.get('enable', False)):
        prefix = {
            'output_dir': str(prefix_cfg.output_dir),
            'num_scenarios': int(prefix_cfg.num_scenarios),
            'dataset_names': {
                condition: str(prefix_cfg.dataset_names[condition])
                for condition in conditions
            },
        }
    marker_appearance = marker_raster.configured_marker_appearance(cfg)
    scenario_sampling = cfg.get('scenario_sampling')
    fixed_protocol = None
    if scenario_sampling is not None:
        fixed_protocol = {
            'version': 1,
            'policy': 'TunnelSearchPolicy',
            'decision_rule': 'cue_aware_nearest_unexplored',
            'validation_profile': 'fixed_layout',
            **(
                {'source_revision': os.environ['STABLEWM_SOURCE_REVISION']}
                if os.environ.get('STABLEWM_SOURCE_REVISION')
                else {}
            ),
        }
    return {
        'environment': 'swm/TunnelRooms-v1',
        # Preserve hashes of existing TR3 journals. Only non-default condition
        # sets need to become part of the persisted collection identity.
        **(
            {'conditions': list(conditions)}
            if conditions != CONDITIONS
            else {}
        ),
        'num_leaves': int(cfg.num_leaves),
        'scenario_seed': int(cfg.scenario_seed),
        'policy_seed': int(cfg.policy_seed),
        'agent_speed': float(cfg.agent_speed),
        **(
            {'marker_appearance': marker_appearance}
            if marker_appearance is not None
            else {}
        ),
        **(
            {'agent_render_radius': float(cfg.agent_render_radius)}
            if cfg.get('agent_render_radius') is not None
            else {}
        ),
        'scenario_constraints': dict(cfg.get('scenario_constraints') or {}),
        **(
            {
                'scenario_sampling': OmegaConf.to_container(
                    scenario_sampling, resolve=True
                ),
                'fixed_layout_protocol': fixed_protocol,
            }
            if scenario_sampling is not None
            else {}
        ),
        'max_episode_steps': int(cfg.world.max_episode_steps),
        'image_shape': [int(item) for item in cfg.world.image_shape],
        'action_noise': float(policy['action_noise']),
        'action_repeat_prob': float(policy['action_repeat_prob']),
        'max_attempts_per_condition': int(cfg.max_attempts_per_condition),
        'dataset_names': {
            condition: str(cfg.dataset_names[condition])
            for condition in conditions
        },
        'prefix': prefix,
    }


def journal_header(cfg: DictConfig, paths: Mapping[str, DatasetFiles]) -> dict:
    """Build the immutable first journal record for this collection."""
    spec = _collection_spec(cfg)
    return {
        'type': 'header',
        'schema_version': SCHEMA_VERSION,
        'collection_spec': spec,
        'collection_spec_hash': marker_raster.sha256_appearance_spec(spec),
        'target_scenarios': int(cfg.num_scenarios),
        'datasets': {
            condition: str(files.final) for condition, files in paths.items()
        },
    }


def rewrite_journal(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomically replace a journal with canonical JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    with temporary.open('w') as file:
        for record in records:
            file.write(canonical_json(record) + '\n')
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def append_journal(path: Path, record: dict[str, Any]) -> None:
    """Durably append one transaction record to an existing journal."""
    with path.open('a') as file:
        file.write(canonical_json(record) + '\n')
        file.flush()
        os.fsync(file.fileno())


def load_journal(
    path: Path, repair_trailing_line: bool = False
) -> list[dict[str, Any]]:
    """Load JSONL, optionally dropping only an interrupted final record."""
    if not path.exists():
        return []

    data = path.read_text()
    lines = data.splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            is_trailing_line = index == len(lines) - 1
            if not repair_trailing_line or not is_trailing_line:
                raise RuntimeError(
                    f'Corrupt collection journal: {path}'
                ) from error
            rewrite_journal(path, records)
            break
    else:
        if repair_trailing_line and lines and not data.endswith('\n'):
            with path.open('a') as file:
                file.write('\n')
                file.flush()
                os.fsync(file.fileno())
    return records


def pair_commits(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract pair commits and require contiguous zero-based episode IDs."""
    commits = [
        record for record in records if record.get('type') == 'pair_commit'
    ]
    for expected_episode, record in enumerate(commits):
        if int(record.get('episode', -1)) != expected_episode:
            raise RuntimeError(
                'Collection journal pair commits are not contiguous.'
            )
    return commits


def validate_commits_against_scenarios(
    commits: Sequence[dict[str, Any]],
    scenarios: Sequence[dict[str, Any]],
) -> None:
    """Require the journal to describe an exact prefix of frozen scenarios."""
    if len(commits) > len(scenarios):
        raise RuntimeError(
            'Journal contains more commits than frozen scenarios.'
        )
    for episode, (commit, scenario) in enumerate(
        zip(commits, scenarios, strict=False)
    ):
        expected = {
            'episode': episode,
            'scenario_hash': scenario['scenario_hash'],
            'target_rank': int(scenario['target_rank']),
            'search_order': list(scenario['search_order']),
        }
        if {key: commit.get(key) for key in expected} != expected:
            raise RuntimeError(
                f'Journal commit {episode} does not match its frozen scenario.'
            )


# -----------------------------------------------------------------------------
# HDF5 append and restart recovery
# -----------------------------------------------------------------------------


def hdf5_counts(path: Path) -> tuple[int, int]:
    """Return episode and row counts after validating canonical offsets."""
    with h5py.File(path, 'r') as file:
        lengths = file['ep_len'][:].astype(np.int64)
        offsets = file['ep_offset'][:].astype(np.int64)
    expected_offsets = (
        np.concatenate(([0], np.cumsum(lengths[:-1])))
        if len(lengths)
        else np.empty(0, dtype=np.int64)
    )
    if not np.array_equal(offsets, expected_offsets):
        raise RuntimeError(f'Non-canonical episode offsets in {path}.')
    return len(lengths), int(lengths.sum())


def truncate_hdf5(path: Path, episodes: int, steps: int) -> None:
    """Truncate uncommitted rows to the journal's last durable boundary."""
    if episodes == 0:
        path.unlink(missing_ok=True)
        return

    with h5py.File(path, 'a') as file:
        missing = sorted({'ep_len', 'ep_offset'} - file.keys())
        if missing:
            raise RuntimeError(
                f'{path} is missing HDF5 transaction metadata: {missing}.'
            )
        for key, dataset in file.items():
            if not dataset.shape:
                continue
            target = episodes if key in {'ep_len', 'ep_offset'} else steps
            if dataset.shape[0] < target:
                raise RuntimeError(
                    f'{path}:{key} is shorter than the committed journal state '
                    f'({dataset.shape[0]} < {target}).'
                )
            if dataset.maxshape is None or dataset.maxshape[0] is not None:
                raise RuntimeError(
                    f'{path}:{key} cannot be transactionally resized.'
                )
            dataset.resize(target, axis=0)
        file.flush()

    if hdf5_counts(path) != (episodes, steps):
        raise RuntimeError(f'{path} could not be restored to journal state.')


def recover_partials(
    paths: Mapping[str, DatasetFiles],
    commits: Sequence[dict[str, Any]],
    target: int,
) -> None:
    """Reconcile each HDF5 file to the last pair committed in the journal."""
    committed = len(commits)
    for condition, files in paths.items():
        path = files.existing()
        if committed < target and files.final.exists():
            raise RuntimeError(
                f'Incomplete journal cannot resume from published file {files.final}.'
            )
        if committed and path is None:
            raise RuntimeError(
                f'Missing {condition} HDF5 for committed journal rows.'
            )
        if path is None:
            continue

        expected_lengths = np.asarray(
            [
                int(record['conditions'][condition]['steps'])
                for record in commits
            ],
            dtype=np.int64,
        )
        truncate_hdf5(path, committed, int(expected_lengths.sum()))
        if committed:
            with h5py.File(path, 'r') as file:
                if not np.array_equal(file['ep_len'][:], expected_lengths):
                    raise RuntimeError(
                        f'{path} episode lengths disagree with the journal.'
                    )


def append_episode(path: Path, episode: dict[str, list[Any]]) -> None:
    """Append one already-canonicalized episode to an HDF5 file."""
    with HDF5Writer(path, mode='append') as writer:
        writer.write_episode(episode)


# -----------------------------------------------------------------------------
# Restartable 10k-to-50k prefix import
# -----------------------------------------------------------------------------


def _copy_prefix_file(source: Path, destination: Path) -> None:
    """Copy through a temporary path so interrupted copies are never active."""
    temporary = destination.with_name(f'.{destination.name}.copying')
    temporary.unlink(missing_ok=True)
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _validate_prefix_source(
    cfg: DictConfig,
    prefix_dir: Path,
    prefix_count: int,
    prefix_manifest: Mapping[str, Any],
    prefix_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Require a completed prefix with matching data-generation settings."""
    audit_path = prefix_dir / AUDIT_MANIFEST_NAME
    if not audit_path.exists():
        raise FileNotFoundError(
            f'Missing completed prefix audit: {audit_path}'
        )
    audit = json.loads(audit_path.read_text())
    conditions = conditions_from_cfg(cfg)
    expected_names = {
        condition: str(cfg.prefix.dataset_names[condition])
        for condition in conditions
    }
    audit_valid = (
        int(audit.get('schema_version', -1)) == SCHEMA_VERSION
        and int(audit.get('num_paired_scenarios', -1)) == prefix_count
        and audit.get('scenario_stream_hash')
        == prefix_manifest.get('stream_hash')
        and audit.get('dataset_names') == expected_names
        and all(
            int(
                audit.get('dataset_summaries', {})
                .get(condition, {})
                .get('num_episodes', -1)
            )
            == prefix_count
            for condition in conditions
        )
    )
    if not audit_valid:
        raise RuntimeError(
            'Prefix audit does not describe a complete dataset.'
        )

    if not prefix_records or prefix_records[0].get('type') != 'header':
        raise RuntimeError('Prefix journal does not begin with a header.')
    prefix_spec = prefix_records[0].get('collection_spec')
    if not isinstance(prefix_spec, dict) or prefix_records[0].get(
        'collection_spec_hash'
    ) != marker_raster.sha256_appearance_spec(prefix_spec):
        raise RuntimeError('Prefix journal has an invalid collection spec.')
    if int(prefix_records[0].get('target_scenarios', -1)) != prefix_count:
        raise RuntimeError(
            'Prefix journal target does not match the boundary.'
        )

    target_spec = _collection_spec(cfg)
    shared_fields = (
        'environment',
        'num_leaves',
        'scenario_seed',
        'policy_seed',
        'agent_speed',
        'scenario_constraints',
        'max_episode_steps',
        'image_shape',
        'action_noise',
        'action_repeat_prob',
    )
    mismatched = [
        field
        for field in shared_fields
        if prefix_spec.get(field) != target_spec.get(field)
    ]
    prefix_appearance = prefix_spec.get(
        'marker_appearance', marker_raster.DEFAULT_MARKER_APPEARANCE
    )
    target_appearance = target_spec.get(
        'marker_appearance', marker_raster.DEFAULT_MARKER_APPEARANCE
    )
    if prefix_appearance != target_appearance:
        mismatched.append('marker_appearance')
    prefix_radius = float(
        prefix_spec.get(
            'agent_render_radius', marker_raster.DEFAULT_AGENT_RENDER_RADIUS
        )
    )
    target_radius = float(
        target_spec.get(
            'agent_render_radius', marker_raster.DEFAULT_AGENT_RENDER_RADIUS
        )
    )
    if prefix_radius != target_radius:
        mismatched.append('agent_render_radius')
    if mismatched:
        raise RuntimeError(
            f'Prefix collection settings differ from target: {mismatched}.'
        )

    prefix_commits = pair_commits(prefix_records)
    complete = any(
        record.get('type') == 'collection_complete'
        and int(record.get('num_paired_scenarios', -1)) == prefix_count
        for record in prefix_records
    )
    if len(prefix_commits) != prefix_count or not complete:
        raise RuntimeError('Prefix journal is not complete at the boundary.')
    validate_commits_against_scenarios(
        prefix_commits,
        prefix_manifest.get('scenarios', [])[:prefix_count],
    )
    return prefix_commits


def initialize_from_prefix(
    cfg: DictConfig,
    scenario_manifest: dict[str, Any],
    paths: Mapping[str, DatasetFiles],
    journal_path: Path,
) -> bool:
    """Idempotently seed a larger collection from its completed 10k prefix."""
    prefix_cfg = cfg.get('prefix')
    if prefix_cfg is None or not bool(prefix_cfg.get('enable', False)):
        return False

    conditions = conditions_from_cfg(cfg)
    prefix_count = int(prefix_cfg.num_scenarios)
    if not 0 < prefix_count < int(cfg.num_scenarios):
        raise ValueError(
            'prefix.num_scenarios must be between zero and target.'
        )

    prefix_dir = Path(str(prefix_cfg.output_dir))
    prefix_manifest_path = prefix_dir / SCENARIO_MANIFEST_NAME
    prefix_scenarios = json.loads(prefix_manifest_path.read_text())
    expected_hashes = [
        row['scenario_hash']
        for row in scenario_manifest['scenarios'][:prefix_count]
    ]
    observed_hashes = [
        row['scenario_hash']
        for row in prefix_scenarios.get('scenarios', [])[:prefix_count]
    ]
    if (
        observed_hashes != expected_hashes
        or len(observed_hashes) != prefix_count
    ):
        raise RuntimeError(
            'The requested 10k prefix is not the same scenario stream.'
        )

    prefix_records = load_journal(prefix_dir / JOURNAL_NAME)
    prefix_commits = _validate_prefix_source(
        cfg,
        prefix_dir,
        prefix_count,
        prefix_scenarios,
        prefix_records,
    )

    header = journal_header(cfg, paths)
    intent = {
        'type': 'prefix_import_intent',
        'prefix_output_dir': str(prefix_dir),
        'prefix_scenarios': prefix_count,
        'prefix_dataset_names': {
            condition: str(prefix_cfg.dataset_names[condition])
            for condition in conditions
        },
    }
    current_records = load_journal(journal_path, repair_trailing_line=True)
    if current_records:
        if current_records[:2] != [header, intent]:
            raise RuntimeError(
                'Prefix import journal does not match this run.'
            )
    else:
        rewrite_journal(journal_path, [header, intent])

    dataset_root = Path(cfg.dataset_cache_dir) / 'datasets'
    for condition in conditions:
        source_name = str(prefix_cfg.dataset_names[condition])
        source = dataset_root / f'{source_name}.h5'
        if not source.exists():
            raise FileNotFoundError(f'Missing prefix dataset: {source}')
        validate_complete_dataset(source, condition, prefix_commits)
        destination = paths[condition].partial
        if source.resolve() == destination.resolve():
            raise ValueError(
                'Prefix source and destination dataset must differ.'
            )
        _copy_prefix_file(source, destination)

    imported_commits = [
        {**record, 'imported_from_prefix': str(prefix_dir)}
        for record in prefix_commits
    ]
    rewrite_journal(
        journal_path,
        [
            header,
            *imported_commits,
            {
                'type': 'prefix_import_complete',
                'prefix_output_dir': str(prefix_dir),
                'prefix_scenarios': prefix_count,
            },
        ],
    )
    return True


# -----------------------------------------------------------------------------
# Canonical episode identity and content hashes
# -----------------------------------------------------------------------------


def canonicalize_episode(
    episode: Mapping[str, Sequence[Any]], episode_index: int, offset: int
) -> dict[str, list[Any]]:
    """Replace row identity columns with contiguous canonical values."""
    if not episode:
        raise RuntimeError('Collector produced an empty episode.')
    lengths = {key: len(values) for key, values in episode.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f'Collector produced mismatched columns: {lengths}')
    steps = next(iter(lengths.values()))
    if steps <= 0:
        raise RuntimeError('Collector produced a zero-step episode.')
    missing = [key for key in TRAINING_KEYS if key not in episode]
    if missing:
        raise RuntimeError(f'Collector omitted training columns: {missing}')

    canonical = {key: list(values) for key, values in episode.items()}
    canonical['id'] = list(np.arange(offset, offset + steps, dtype=np.int64))
    canonical['ep_idx'] = [np.int32(episode_index)] * steps
    canonical['step_idx'] = list(np.arange(steps, dtype=np.int64))
    return canonical


def episode_sha256(episode: Mapping[str, Any]) -> str:
    """Hash canonical episode contents independent of Python containers."""
    digest = hashlib.sha256()
    for key in sorted(episode):
        values = np.asarray(episode[key])
        digest.update(key.encode('utf-8') + b'\0')
        digest.update(values.dtype.str.encode('ascii') + b'\0')
        digest.update(
            canonical_json(list(values.shape)).encode('ascii') + b'\0'
        )
        if values.dtype.kind in {'O', 'S', 'U'}:
            digest.update(canonical_json(values.tolist()).encode('utf-8'))
        else:
            digest.update(np.ascontiguousarray(values).tobytes())
        digest.update(b'\0')
    return digest.hexdigest()


# -----------------------------------------------------------------------------
# Complete validation and publication
# -----------------------------------------------------------------------------


def validate_complete_dataset(
    path: Path,
    condition: str,
    commits: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate every row and content hash against the transaction journal."""
    episodes, steps = hdf5_counts(path)
    expected_lengths = np.asarray(
        [int(record['conditions'][condition]['steps']) for record in commits],
        dtype=np.int64,
    )
    if episodes != len(commits) or steps != int(expected_lengths.sum()):
        raise RuntimeError(f'{path} does not match the committed journal.')

    with h5py.File(path, 'r') as file:
        missing = [
            key
            for key in (*TRAINING_KEYS, 'id', 'ep_idx', 'step_idx')
            if key not in file
        ]
        if missing:
            raise RuntimeError(
                f'{path} is missing canonical columns: {missing}'
            )

        lengths = file['ep_len'][:].astype(np.int64)
        offsets = file['ep_offset'][:].astype(np.int64)
        if not np.array_equal(lengths, expected_lengths):
            raise RuntimeError(
                f'{path} episode lengths do not match the journal.'
            )
        if not np.array_equal(file['id'][:], np.arange(steps, dtype=np.int64)):
            raise RuntimeError(f'{path} has non-canonical row ids.')

        expected_ep_idx = np.repeat(
            np.arange(episodes, dtype=np.int64), lengths
        )
        if not np.array_equal(file['ep_idx'][:], expected_ep_idx):
            raise RuntimeError(f'{path} has non-canonical episode ids.')
        expected_step_idx = np.concatenate(
            [np.arange(length, dtype=np.int64) for length in lengths]
        )
        if not np.array_equal(file['step_idx'][:], expected_step_idx):
            raise RuntimeError(f'{path} has non-canonical step ids.')

        terminal_rows = offsets + lengths - 1
        if not np.all(file['terminated'][terminal_rows].astype(bool)):
            raise RuntimeError(
                f'{path} contains a non-successful final episode.'
            )
        if np.any(file['truncated'][terminal_rows].astype(bool)):
            raise RuntimeError(f'{path} contains a truncated final episode.')

        data_keys = sorted(set(file) - {'ep_len', 'ep_offset'})
        observed_hashes: list[str] = []
        for episode_index, (offset, length) in enumerate(
            zip(offsets, lengths, strict=True)
        ):
            start = int(offset)
            stop = start + int(length)
            observed_hash = episode_sha256(
                {key: file[key][start:stop] for key in data_keys}
            )
            expected_hash = commits[episode_index]['conditions'][
                condition
            ].get('episode_sha256')
            if expected_hash is None:
                raise RuntimeError(
                    f'Journal commit {episode_index} lacks '
                    f'{condition} episode hash.'
                )
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f'{path} episode {episode_index} content does not match '
                    'the committed journal.'
                )
            observed_hashes.append(observed_hash)

    content_digest = hashlib.sha256()
    for episode_hash in observed_hashes:
        content_digest.update(episode_hash.encode('ascii'))
    return {
        'num_episodes': episodes,
        'num_steps': steps,
        'content_sha256': content_digest.hexdigest(),
        'file_size_bytes': path.stat().st_size,
        'episode_lengths': lengths,
    }


def publish_completed(
    paths: Mapping[str, DatasetFiles],
    commits: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate the complete pair before publishing either partial file."""
    validations: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    for condition, files in paths.items():
        source = files.existing()
        if source is None:
            raise FileNotFoundError(
                f'Missing completed dataset for {condition}: {files.final}'
            )
        sources[condition] = source
        validations[condition] = validate_complete_dataset(
            source, condition, commits
        )

    for condition, files in paths.items():
        source = sources[condition]
        if source == files.partial:
            os.replace(files.partial, files.final)
    return validations


# -----------------------------------------------------------------------------
# Audit summaries and explicit cleanup
# -----------------------------------------------------------------------------


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    """Return stable descriptive statistics, including all tied modes."""
    array = np.asarray(values)
    unique, counts = np.unique(array, return_counts=True)
    modes = unique[counts == counts.max()]
    return {
        'mean': float(np.mean(array)),
        'median': float(np.median(array)),
        'mode': float(modes[0]),
        'modes': [float(value) for value in modes],
        'min': float(np.min(array)),
        'max': float(np.max(array)),
    }


def summarize_dataset(
    path: Path,
    condition: str,
    commits: Sequence[dict[str, Any]],
    scenarios: Sequence[dict[str, Any]],
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build audit-ready length, rank, retry, and action statistics."""
    checked = dict(
        validation or validate_complete_dataset(path, condition, commits)
    )
    lengths = np.asarray(checked.pop('episode_lengths'), dtype=np.int64)
    condition_rows = [record['conditions'][condition] for record in commits]
    ranks = np.asarray(
        [scenario['target_rank'] for scenario in scenarios], dtype=np.int64
    )
    retries = np.asarray(
        [record['retries'] for record in condition_rows], dtype=np.int64
    )
    retry_by_stratum = {}
    if scenarios and all('start_stratum' in row for row in scenarios):
        strata = sorted({row['start_stratum'] for row in scenarios})
        retry_by_stratum = {
            stratum: numeric_summary(
                retries[
                    np.asarray(
                        [row['start_stratum'] == stratum for row in scenarios],
                        dtype=bool,
                    )
                ]
            )
            for stratum in strata
        }
    return {
        **checked,
        'episode_length': numeric_summary(lengths),
        'target_rank_counts': {
            str(rank): int(np.sum(ranks == rank))
            for rank in range(1, int(max(ranks)) + 1)
        },
        'retry': numeric_summary(retries),
        **(
            {'retry_by_start_stratum': retry_by_stratum}
            if retry_by_stratum
            else {}
        ),
        'action_clip_fraction_mean': float(
            np.mean(
                [record['action_clip_fraction'] for record in condition_rows]
            )
        ),
        'consecutive_action_repeat_rate_mean': float(
            np.mean(
                [
                    record['consecutive_action_repeat_rate']
                    for record in condition_rows
                ]
            )
        ),
    }


def reset_outputs(output_dir: Path, paths: Mapping[str, DatasetFiles]) -> None:
    """Delete this collection's datasets and manifests for an explicit reset."""
    for files in paths.values():
        files.final.unlink(missing_ok=True)
        files.partial.unlink(missing_ok=True)
    for name in (
        SCENARIO_MANIFEST_NAME,
        AUDIT_MANIFEST_NAME,
        EVAL_MANIFEST_NAME,
        JOURNAL_NAME,
    ):
        (output_dir / name).unlink(missing_ok=True)


__all__ = [
    'AUDIT_MANIFEST_NAME',
    'CONDITIONS',
    'DatasetFiles',
    'DatasetPaths',
    'EVAL_MANIFEST_NAME',
    'JOURNAL_NAME',
    'SCENARIO_MANIFEST_NAME',
    'SCHEMA_VERSION',
    'TRAINING_KEYS',
    'append_episode',
    'append_journal',
    'atomic_write_json',
    'canonical_json',
    'canonicalize_episode',
    'conditions_from_cfg',
    'dataset_paths',
    'episode_sha256',
    'hdf5_counts',
    'initialize_from_prefix',
    'journal_header',
    'load_journal',
    'numeric_summary',
    'pair_commits',
    'policy_settings',
    'publish_completed',
    'recover_partials',
    'reset_outputs',
    'rewrite_journal',
    'summarize_dataset',
    'truncate_hdf5',
    'validate_commits_against_scenarios',
    'validate_complete_dataset',
]
