"""Orchestrate resumable manifest-first Tunnel Rooms collection.

This module deliberately contains only the collection workflow. The durable
scenario stream and HDF5 transaction logic live in two focused neighboring
modules. The order of operations stays easy to audit: freeze scenarios,
recover committed state, collect missing pairs, validate both datasets, and
publish them together.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig

import stable_worldmodel as swm
import scripts.data.tunnel_rooms_paired_io as collection_io
from scripts.data.tunnel_rooms_paired_scenarios import (
    build_expert_policy,
    freeze_scenario_manifest,
)


# -----------------------------------------------------------------------------
# Noisy paired rollout execution
# -----------------------------------------------------------------------------


class EpisodeCapture:
    """Buffered writer implementing the protocol expected by ``World.collect``."""

    def __init__(self) -> None:
        self.episode: dict[str, list[Any]] | None = None

    def __enter__(self) -> 'EpisodeCapture':
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def write_episodes(self, episodes: Iterable[dict[str, list[Any]]]) -> None:
        self.episode = next(iter(episodes))


def noise_seed_for(
    policy_seed: int, episode: int, condition: str, attempt: int
) -> int:
    """Derive a stable, condition-specific retry seed."""
    payload = f'{policy_seed}:{episode}:{condition}:{attempt}'.encode('ascii')
    return (
        int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')
        & 0x7FFFFFFF
    )


def _terminal_status(episode: dict[str, list[Any]]) -> tuple[bool, str]:
    """Return environmental success and its terminal-state label.

    Gymnasium's time-limit wrapper may mark the final transition as both
    terminated and truncated when the agent succeeds on the last allowed step.
    The flags describe independent causes, so retain that distinction in the
    retry audit. Accepted episodes are normalized before persistence.
    """
    terminated = bool(np.asarray(episode['terminated'][-1]).squeeze())
    truncated = bool(np.asarray(episode['truncated'][-1]).squeeze())
    status = (
        'terminated_at_time_limit'
        if terminated and truncated
        else 'terminated'
        if terminated
        else 'truncated'
        if truncated
        else 'incomplete'
    )
    return terminated, status


def _attempt_trace(
    world: Any,
    episode: dict[str, list[Any]],
    attempt: int,
    noise_seed: int,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Summarize one retry using terminal, route, and action diagnostics."""
    success, status = _terminal_status(episode)
    metadata = world.envs.envs[0].unwrapped.audit_metadata()
    actions = np.asarray(episode['action'], dtype=np.float32)
    repeat_rate = 0.0
    if len(actions) > 1:
        repeat_rate = float(
            np.mean(np.all(actions[1:] == actions[:-1], axis=-1))
        )
    return {
        'attempt': int(attempt),
        'noise_seed': int(noise_seed),
        'success': success,
        'terminal_status': status,
        'steps': len(next(iter(episode.values()))),
        'teleport_count': int(metadata.get('teleport_count', 0)),
        'route_length': float(metadata.get('route_length', 0.0)),
        'hub_entry_trace': [
            int(endpoint) for endpoint in metadata.get('hub_entry_trace', [])
        ],
        'action_clip_fraction': float(np.mean(np.abs(actions) >= 0.999)),
        'consecutive_action_repeat_rate': repeat_rate,
        'noise': {
            'action_noise_std': float(cfg.policy.action_noise),
            'action_repeat_prob': float(cfg.policy.action_repeat_prob),
        },
    }


def _expected_hub_entries(
    scenario: dict[str, Any], condition: str
) -> list[int]:
    """Return the certified tunnel-entry sequence for one condition."""
    if condition == 'hidden':
        rank = int(scenario['target_rank'])
        return list(scenario['search_order'][:rank])
    return [int(scenario['correct_hub_endpoint'])]


def collect_successful_condition(
    world: Any,
    scenario: dict[str, Any],
    condition: str,
    cfg: DictConfig,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Retry one frozen scenario until success follows its certified route."""
    episode_index = int(scenario['episode'])
    replay = deepcopy(scenario)
    replay.pop('scenario_hash', None)
    replay.pop('episode', None)
    replay['identity_mode'] = condition
    expected_entries = _expected_hub_entries(scenario, condition)
    traces: list[dict[str, Any]] = []

    for attempt in range(1, int(cfg.max_attempts_per_condition) + 1):
        noise_seed = noise_seed_for(
            int(cfg.policy_seed), episode_index, condition, attempt
        )
        world.set_policy(build_expert_policy(condition, noise_seed, cfg))
        capture = EpisodeCapture()
        world.collect(
            writer=capture,
            episodes=1,
            seed=int(scenario['scenario_seed']),
            options={'scenario': replay},
            progress=False,
        )
        if capture.episode is None:
            raise RuntimeError('World.collect did not emit an episode.')

        trace = _attempt_trace(
            world, capture.episode, attempt, noise_seed, cfg
        )
        trace['route_valid'] = trace['hub_entry_trace'] == expected_entries
        traces.append(trace)
        if trace['success'] and trace['route_valid']:
            if trace['terminal_status'] == 'terminated_at_time_limit':
                capture.episode['truncated'][-1] = np.zeros_like(
                    capture.episode['truncated'][-1], dtype=bool
                )
            return capture.episode, {
                'accepted_attempt': attempt,
                'accepted_noise_seed': noise_seed,
                'retries': attempt - 1,
                'steps': int(trace['steps']),
                'terminal_status': trace['terminal_status'],
                'teleport_count': trace['teleport_count'],
                'route_length': trace['route_length'],
                'hub_entry_trace': trace['hub_entry_trace'],
                'action_clip_fraction': trace['action_clip_fraction'],
                'consecutive_action_repeat_rate': trace[
                    'consecutive_action_repeat_rate'
                ],
                'traces': traces,
            }

    raise RuntimeError(
        f'{condition} failed scenario {episode_index} after '
        f'{int(cfg.max_attempts_per_condition)} deterministic attempts.'
    )


def _make_world(condition: str, cfg: DictConfig) -> Any:
    """Create one single-environment collector world."""
    world_kwargs = dict(cfg.world)
    world_kwargs['num_envs'] = 1
    return swm.World(
        'swm/TunnelRooms-v1',
        **world_kwargs,
        render_mode='rgb_array',
        identity_mode=condition,
        num_leaves=int(cfg.num_leaves),
    )


@contextmanager
def _open_collection_worlds(cfg: DictConfig) -> Iterator[dict[str, Any]]:
    """Open every requested condition world and close it on every exit path."""
    with ExitStack() as resources:
        worlds = {}
        for condition in collection_io.conditions_from_cfg(cfg):
            world = _make_world(condition, cfg)
            resources.callback(world.close)
            worlds[condition] = world
        yield worlds


def _audit_rows(
    scenarios: Sequence[dict[str, Any]],
    commits: Sequence[dict[str, Any]],
    conditions: Sequence[str],
) -> list[dict[str, Any]]:
    """Flatten paired commits into one audit row per render condition."""
    rows: list[dict[str, Any]] = []
    for scenario, commit in zip(scenarios, commits, strict=True):
        if scenario['scenario_hash'] != commit['scenario_hash']:
            raise RuntimeError(
                'Journal scenario hash does not match frozen manifest.'
            )
        for condition in conditions:
            row = deepcopy(scenario)
            row.update(
                {
                    'condition': condition,
                    'identity_mode': condition,
                    'success': True,
                    'timeout': False,
                    'truncated': False,
                    **deepcopy(commit['conditions'][condition]),
                }
            )
            rows.append(row)
    return rows


def build_audit_manifest(
    cfg: DictConfig,
    output_dir: Path,
    scenario_manifest: Mapping[str, Any],
    paths: Mapping[str, collection_io.DatasetFiles],
    commits: Sequence[dict[str, Any]],
    dataset_summaries: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Describe collection inputs, outputs, retries, and transaction rules."""
    conditions = collection_io.conditions_from_cfg(cfg)
    scenarios = scenario_manifest['scenarios']
    scenario_distribution: dict[str, Any] = {}
    if cfg.get('scenario_sampling') is not None:
        endpoints = scenarios[0]['endpoint_positions']
        if any(row['endpoint_positions'] != endpoints for row in scenarios):
            raise RuntimeError(
                'Fixed-layout collection changed endpoint geometry.'
            )
        scenario_distribution = {
            'fixed_endpoint_positions': endpoints,
            'geometry_sha256': hashlib.sha256(
                collection_io.canonical_json(endpoints).encode('utf-8')
            ).hexdigest(),
            'start_strata': dict(
                sorted(
                    Counter(row['start_stratum'] for row in scenarios).items()
                )
            ),
            'routing_permutations': dict(
                sorted(
                    Counter(str(row['pairing']) for row in scenarios).items()
                )
            ),
            'cue_assignments': dict(
                sorted(
                    Counter(
                        ','.join(map(str, row['cue_assignment']))
                        for row in scenarios
                    ).items()
                )
            ),
            'target_ranks': dict(
                sorted(
                    Counter(
                        str(row['target_rank']) for row in scenarios
                    ).items()
                )
            ),
            'joint_counts': dict(
                sorted(
                    Counter(
                        '|'.join(
                            (
                                row['start_stratum'],
                                f'route={row["pairing"]}',
                                'cue='
                                + ','.join(map(str, row['cue_assignment'])),
                                f'target={row["target_leaf"]}',
                                f'rank={row["target_rank"]}',
                            )
                        )
                        for row in scenarios
                    ).items()
                )
            ),
        }
    return {
        'schema_version': collection_io.SCHEMA_VERSION,
        'environment': 'swm/TunnelRooms-v1',
        'num_leaves': int(cfg.num_leaves),
        'conditions': list(conditions),
        'dataset_names': {
            condition: str(cfg.dataset_names[condition])
            for condition in conditions
        },
        'training_keys': list(collection_io.TRAINING_KEYS),
        'num_collected_scenarios': len(commits),
        **(
            {'num_paired_scenarios': len(commits)}
            if len(conditions) > 1
            else {}
        ),
        'datasets': {
            condition: str(files.final) for condition, files in paths.items()
        },
        'dataset_summaries': dict(dataset_summaries),
        'scenario_manifest': str(
            output_dir / collection_io.SCENARIO_MANIFEST_NAME
        ),
        'scenario_stream_hash': scenario_manifest['stream_hash'],
        **(
            {
                'fixed_layout_protocol': {
                    'version': 1,
                    'policy': 'TunnelSearchPolicy',
                    'decision_rule': 'cue_aware_nearest_unexplored',
                    'validation_profile': 'fixed_layout',
                    **(
                        {
                            'source_revision': os.environ[
                                'STABLEWM_SOURCE_REVISION'
                            ]
                        }
                        if os.environ.get('STABLEWM_SOURCE_REVISION')
                        else {}
                    ),
                }
            }
            if cfg.get('scenario_sampling') is not None
            else {}
        ),
        **(
            {'scenario_distribution': scenario_distribution}
            if scenario_distribution
            else {}
        ),
        'journal': str(output_dir / collection_io.JOURNAL_NAME),
        'episodes': _audit_rows(
            scenario_manifest['scenarios'], commits, conditions
        ),
        'collection_policy': {
            **{condition: 'TunnelSearchPolicy' for condition in conditions},
            **collection_io.policy_settings(cfg),
            'max_attempts_per_condition': int(cfg.max_attempts_per_condition),
            'same_scenario_retries': True,
            'deterministic_condition_attempt_seeds': True,
            'raw_oversampling': False,
            'rank_filtering': False,
        },
        'transaction': {
            'partial_suffix': '.partial.h5',
            'pair_commit_journal': str(
                output_dir / collection_io.JOURNAL_NAME
            ),
            'journal_is_authoritative': True,
        },
    }


def _prepare_outputs(
    cfg: DictConfig,
    output_dir: Path,
    paths: collection_io.DatasetPaths,
    journal_path: Path,
) -> None:
    """Apply the explicit overwrite/resume policy before reading state."""
    if bool(cfg.get('overwrite', False)):
        collection_io.reset_outputs(output_dir, paths)
        return
    if bool(cfg.get('resume', True)):
        return

    occupied = [
        path
        for files in paths.values()
        for path in (files.final, files.partial)
        if path.exists()
    ]
    if occupied or journal_path.exists():
        raise FileExistsError('Collection outputs exist and resume=false.')


def _load_collection_state(
    cfg: DictConfig,
    scenario_manifest: Mapping[str, Any],
    paths: collection_io.DatasetPaths,
    journal_path: Path,
) -> list[dict[str, Any]]:
    """Initialize or recover the journal-authoritative committed prefix."""
    records = collection_io.load_journal(
        journal_path, repair_trailing_line=True
    )
    existing_datasets = any(
        path.exists()
        for files in paths.values()
        for path in (files.final, files.partial)
    )

    if not records:
        if existing_datasets:
            raise RuntimeError(
                'HDF5 output exists without a transaction journal.'
            )
        imported = collection_io.initialize_from_prefix(
            cfg, scenario_manifest, paths, journal_path
        )
        if not imported:
            collection_io.rewrite_journal(
                journal_path, [collection_io.journal_header(cfg, paths)]
            )
        records = collection_io.load_journal(journal_path)
    elif any(
        record.get('type') == 'prefix_import_intent' for record in records
    ):
        if any(
            record.get('type') == 'prefix_import_complete'
            for record in records
        ):
            raise RuntimeError(
                'Completed prefix journal retained an import intent.'
            )
        collection_io.initialize_from_prefix(
            cfg, scenario_manifest, paths, journal_path
        )
        records = collection_io.load_journal(journal_path)

    expected = collection_io.journal_header(cfg, paths)
    if not records or records[0].get('type') != 'header':
        raise RuntimeError('Collection journal does not begin with a header.')
    if records[0].get('collection_spec_hash') != expected[
        'collection_spec_hash'
    ] or int(records[0].get('target_scenarios', -1)) != int(cfg.num_scenarios):
        raise RuntimeError(
            'Collection journal configuration does not match this run.'
        )

    commits = collection_io.pair_commits(records)
    collection_io.validate_commits_against_scenarios(
        commits, scenario_manifest['scenarios']
    )
    collection_io.recover_partials(paths, commits, int(cfg.num_scenarios))
    if len(commits) > int(cfg.num_scenarios):
        raise RuntimeError('Journal contains more pairs than requested.')
    return commits


def _collect_missing_pairs(
    cfg: DictConfig,
    scenarios: Sequence[dict[str, Any]],
    paths: collection_io.DatasetPaths,
    journal_path: Path,
    commits: list[dict[str, Any]],
) -> None:
    """Append uncommitted scenarios, then durably journal each transaction."""
    conditions = collection_io.conditions_from_cfg(cfg)
    offsets = {
        condition: sum(
            int(record['conditions'][condition]['steps']) for record in commits
        )
        for condition in conditions
    }

    with _open_collection_worlds(cfg) as worlds:
        for scenario in scenarios[len(commits) :]:
            accepted: dict[str, dict[str, Any]] = {}
            condition_records: dict[str, dict[str, Any]] = {}
            for condition in conditions:
                episode, record = collect_successful_condition(
                    worlds[condition], scenario, condition, cfg
                )
                canonical_episode = collection_io.canonicalize_episode(
                    episode,
                    int(scenario['episode']),
                    offsets[condition],
                )
                record['episode_sha256'] = collection_io.episode_sha256(
                    canonical_episode
                )
                accepted[condition] = canonical_episode
                condition_records[condition] = record

            # Commit only after every requested HDF5 append succeeds.
            for condition in conditions:
                collection_io.append_episode(
                    paths[condition].partial, accepted[condition]
                )
            commit = {
                'type': 'pair_commit',
                'episode': int(scenario['episode']),
                'scenario_seed': int(scenario['scenario_seed']),
                'scenario_hash': scenario['scenario_hash'],
                'target_rank': int(scenario['target_rank']),
                'search_order': list(scenario.get('search_order', [])),
                'conditions': condition_records,
            }
            collection_io.append_journal(journal_path, commit)
            commits.append(commit)
            for condition in conditions:
                offsets[condition] += int(
                    condition_records[condition]['steps']
                )

            log_every = int(cfg.get('log_every', 100))
            if len(commits) % log_every == 0:
                logging.info(
                    f'Committed {len(commits)}/{int(cfg.num_scenarios)} '
                    'Tunnel Rooms scenarios.'
                )


def _finish_collection(
    cfg: DictConfig,
    output_dir: Path,
    scenario_manifest: Mapping[str, Any],
    paths: collection_io.DatasetPaths,
    journal_path: Path,
    commits: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate every dataset, publish it, and write the final audit."""
    conditions = collection_io.conditions_from_cfg(cfg)
    if len(commits) != int(cfg.num_scenarios):
        raise RuntimeError(
            'Collection ended before every frozen scenario committed.'
        )

    validations = collection_io.publish_completed(paths, commits)
    summaries = {
        condition: collection_io.summarize_dataset(
            files.final,
            condition,
            commits,
            scenario_manifest['scenarios'],
            validations[condition],
        )
        for condition, files in paths.items()
    }
    manifest = build_audit_manifest(
        cfg,
        output_dir,
        scenario_manifest,
        paths,
        commits,
        summaries,
    )
    collection_io.atomic_write_json(
        output_dir / collection_io.AUDIT_MANIFEST_NAME, manifest
    )

    records = collection_io.load_journal(journal_path)
    if not any(
        record.get('type') == 'collection_complete' for record in records
    ):
        collection_io.append_journal(
            journal_path,
            {
                'type': 'collection_complete',
                'num_collected_scenarios': len(commits),
                **(
                    {'num_paired_scenarios': len(commits)}
                    if len(conditions) > 1
                    else {}
                ),
                'audit_manifest': str(
                    output_dir / collection_io.AUDIT_MANIFEST_NAME
                ),
            },
        )
    logging.success(
        f'Published {len(commits)} successful Tunnel Rooms scenarios.'
    )
    return manifest


def collect_manifest_first_datasets(cfg: DictConfig) -> dict[str, Any]:
    """Run deterministic, resumable collection for requested conditions."""
    num_leaves = int(cfg.num_leaves)
    conditions = collection_io.conditions_from_cfg(cfg)
    if num_leaves == 1 and conditions != ('hidden',):
        raise ValueError(
            'TR1 collection supports exactly conditions=[hidden].'
        )
    if num_leaves == 1 and bool(cfg.prefix.enable):
        raise ValueError('TR1 prefix extension is not supported.')
    fixed_layout = cfg.get('scenario_sampling') is not None
    if (
        num_leaves == 3
        and not fixed_layout
        and (conditions != collection_io.CONDITIONS)
    ):
        raise ValueError(
            'TR3 collection requires matched hidden/color conditions.'
        )
    if fixed_layout and len(conditions) != 1:
        raise ValueError(
            'Fixed-layout datasets must collect exactly one condition per job.'
        )
    if fixed_layout and bool(cfg.prefix.enable):
        raise ValueError('Fixed-layout prefix extension is not supported.')
    if num_leaves == 4 and conditions != ('paired_color',):
        raise ValueError(
            'TR4 collection currently supports exactly '
            'conditions=[paired_color].'
        )
    if num_leaves not in (1, 2, 3, 4):
        raise ValueError(
            'Manifest-first collection supports TR1, TR2, TR3, or TR4.'
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = collection_io.dataset_paths(cfg)
    journal_path = output_dir / collection_io.JOURNAL_NAME
    _prepare_outputs(cfg, output_dir, paths, journal_path)

    scenario_manifest = freeze_scenario_manifest(cfg, output_dir)
    commits = _load_collection_state(
        cfg, scenario_manifest, paths, journal_path
    )
    _collect_missing_pairs(
        cfg,
        scenario_manifest['scenarios'],
        paths,
        journal_path,
        commits,
    )
    return _finish_collection(
        cfg,
        output_dir,
        scenario_manifest,
        paths,
        journal_path,
        commits,
    )


__all__ = ['collect_manifest_first_datasets']
