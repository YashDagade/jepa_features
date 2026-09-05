"""Collect matched hidden and color Tunnel Rooms datasets.

This is the single entry point for Tunnel Rooms collection. Legacy collection
writes each condition through ``World.collect`` and records an out-of-band
audit manifest. Manifest-first modes delegate to the transactional paired
collector without changing the command-line interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig

import stable_worldmodel as swm
from stable_worldmodel.envs.tunnel_rooms import TunnelRoomsEnv
from stable_worldmodel.policy import RandomPolicy
from scripts.data.tunnel_rooms_paired_scenarios import (
    audit_active_rollout,
    audit_scenario_rollout,
    build_expert_policy,
    build_stratified_eval_manifest,
)
from scripts.data.tunnel_rooms_paired_io import policy_settings


TRAINING_KEYS = ['pixels', 'action']
CONDITIONS = ('hidden', 'paired_color')
DATASET_ENV_NAME = 'TunnelRooms'
DATASET_NAMES = {
    'hidden': 'tunnel_room',
    'paired_color': 'tunnel_room_color',
}


# ---------------------------------------------------------------------------
# Collection configuration
# ---------------------------------------------------------------------------


def dataset_names_from_cfg(cfg: DictConfig) -> dict[str, str]:
    """Return the dataset basename for each condition."""
    names = dict(DATASET_NAMES)
    if cfg.get('dataset_names') is not None:
        names.update(
            {key: str(value) for key, value in cfg.dataset_names.items()}
        )
    return names


def reset_options_from_cfg(cfg: DictConfig) -> dict[str, Any] | None:
    """Build reset options shared by collection, auditing, and videos."""
    options: dict[str, Any] = {}
    if cfg.get('agent_speed') is not None:
        options['variation_values'] = {
            'agent.speed': np.asarray([float(cfg.agent_speed)], np.float32)
        }
    if cfg.get('scenario_constraints') is not None:
        options['scenario_constraints'] = {
            key: float(value)
            for key, value in cfg.scenario_constraints.items()
            if value is not None
        }
    if cfg.get('num_leaves') is not None:
        options['scenario_seed_base'] = int(cfg.scenario_seed)
    return options or None


def episode_seeds(initial_seed: int, episodes: int) -> list[int]:
    """Match the single-environment seed schedule in ``World.collect``."""
    return [initial_seed + episode for episode in range(episodes)]


def recording_seed(initial_seed: int, previously_recorded: int) -> int:
    """Continue the ``World.collect`` seed schedule when resuming."""
    return initial_seed + previously_recorded


# ---------------------------------------------------------------------------
# Legacy HDF5 collection
# ---------------------------------------------------------------------------


def ensure_replay_columns(
    dataset_path: Path,
    env_name: str = DATASET_ENV_NAME,
) -> None:
    """Add canonical row metadata used by dataset-driven evaluation."""
    with h5py.File(dataset_path, 'a') as dataset_file:
        episode_lengths = dataset_file['ep_len'][:].astype(np.int64)
        total_steps = int(episode_lengths.sum())
        step_index = (
            np.concatenate(
                [
                    np.arange(length, dtype=np.int64)
                    for length in episode_lengths
                ]
            )
            if total_steps
            else np.empty(0, dtype=np.int64)
        )
        episode_index = np.repeat(
            np.arange(len(episode_lengths), dtype=np.int32),
            episode_lengths,
        )
        env_names = np.asarray([env_name] * total_steps, dtype=object)

        _write_dataset(dataset_file, 'step_idx', step_index)
        _write_dataset(dataset_file, 'ep_idx', episode_index)
        _write_dataset(
            dataset_file,
            'env_name',
            env_names,
            dtype=h5py.string_dtype(encoding='utf-8'),
        )


def record_condition(
    output_dir: Path,
    condition: str,
    cfg: DictConfig,
    dataset_name: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Collect one HDF5 condition without privileged scenario columns."""
    dataset_name = dataset_name or f'tunnel_rooms_{condition}_pilot'
    cache_dir = output_dir if cache_dir is None else cache_dir
    dataset_path = cache_dir / 'datasets' / f'{dataset_name}.h5'
    previously_recorded = 0

    if dataset_path.exists():
        if cfg.overwrite:
            dataset_path.unlink()
        elif not cfg.get('resume', False):
            raise FileExistsError(
                f'{dataset_path} exists; set overwrite=true to replace it or '
                'resume=true to continue it.'
            )
        else:
            with h5py.File(dataset_path, 'r') as dataset_file:
                if 'ep_len' in dataset_file:
                    previously_recorded = int(dataset_file['ep_len'].shape[0])

    remaining = int(cfg.num_scenarios) - previously_recorded
    if remaining <= 0:
        ensure_replay_columns(dataset_path)
        logging.info(
            f'{dataset_path} already contains {previously_recorded} episodes.'
        )
        return dataset_path

    world = _world_for(condition, cfg)
    try:
        world.set_policy(build_expert_policy(condition, cfg.policy_seed, cfg))
        world.collect(
            dataset_path,
            episodes=remaining,
            seed=recording_seed(cfg.scenario_seed, previously_recorded),
            options=reset_options_from_cfg(cfg),
            format='hdf5',
        )
    finally:
        world.close()

    ensure_replay_columns(dataset_path)
    return dataset_path


# ---------------------------------------------------------------------------
# Collection-level rollout auditing
# ---------------------------------------------------------------------------


def audit_seeded_scenarios(
    cfg: DictConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the deterministic legacy stream with both expert policies."""
    envs = {
        condition: TunnelRoomsEnv(
            identity_mode=condition,
            num_leaves=int(cfg.get('num_leaves', 2)),
        )
        for condition in CONDITIONS
    }
    policies = {
        condition: build_expert_policy(condition, cfg.policy_seed, cfg)
        for condition in CONDITIONS
    }
    for condition, policy in policies.items():
        policy.set_env(envs[condition])

    audit: list[dict[str, Any]] = []
    first_scenario: dict[str, Any] | None = None
    try:
        for episode, scenario_seed in enumerate(
            episode_seeds(cfg.scenario_seed, cfg.num_scenarios)
        ):
            obs, info = envs['hidden'].reset(
                seed=scenario_seed,
                options=reset_options_from_cfg(cfg),
            )
            scenario = {
                **envs['hidden'].scenario_spec(),
                'episode': episode,
            }
            if first_scenario is None:
                first_scenario = scenario
            audit.append(
                audit_active_rollout(
                    envs['hidden'],
                    obs,
                    info,
                    scenario,
                    'hidden',
                    cfg.world.max_episode_steps,
                    policies['hidden'],
                )
            )
            audit.append(
                audit_scenario_rollout(
                    scenario,
                    'paired_color',
                    cfg.world.max_episode_steps,
                    env=envs['paired_color'],
                    policy=policies['paired_color'],
                )
            )
    finally:
        for env in envs.values():
            env.close()

    if first_scenario is None:
        raise ValueError('Full collection requires at least one scenario.')
    return audit, first_scenario


def audit_from_stored_datasets(
    cfg: DictConfig,
    paths: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build noisy audit rows from HDF5 terminals and deterministic scenarios.

    Replaying noisy policies is not a reliable rollout-length oracle because
    their RNG streams can diverge. The stored terminal flags therefore remain
    the source of truth for filtering paired successful trajectories.
    """
    status = {
        condition: _dataset_episode_status(path)
        for condition, path in paths.items()
    }
    for condition, values in status.items():
        observed = int(values['lengths'].shape[0])
        if observed != int(cfg.num_scenarios):
            raise RuntimeError(
                f'{condition} dataset has {observed} episodes; '
                f'expected {int(cfg.num_scenarios)}.'
            )

    audit: list[dict[str, Any]] = []
    first_scenario: dict[str, Any] | None = None
    env = TunnelRoomsEnv(
        identity_mode='hidden',
        num_leaves=int(cfg.get('num_leaves', 2)),
    )
    try:
        for episode, scenario_seed in enumerate(
            episode_seeds(cfg.scenario_seed, cfg.num_scenarios)
        ):
            env.reset(
                seed=scenario_seed,
                options=reset_options_from_cfg(cfg),
            )
            scenario = {**env.scenario_spec(), 'episode': episode}
            if first_scenario is None:
                first_scenario = scenario
            for condition in CONDITIONS:
                values = status[condition]
                truncated = bool(values['truncated'][episode])
                audit.append(
                    {
                        **scenario,
                        'condition': condition,
                        'identity_mode': condition,
                        'scenario_seed': scenario_seed,
                        'success': bool(values['terminated'][episode]),
                        'timeout': truncated,
                        'truncated': truncated,
                        'steps': int(values['lengths'][episode]),
                    }
                )
    finally:
        env.close()

    if first_scenario is None:
        raise ValueError('Full collection requires at least one scenario.')
    return audit, first_scenario


def validate_datasets_match_audit(
    paths: dict[str, str],
    audit: list[dict[str, Any]],
) -> None:
    """Require stored episode lengths to match the replay audit."""
    for condition, path in paths.items():
        expected = np.asarray(
            [
                row['steps']
                for row in sorted(
                    (row for row in audit if row['condition'] == condition),
                    key=lambda row: row['episode'],
                )
            ],
            dtype=np.int64,
        )
        with h5py.File(path, 'r') as dataset_file:
            observed = dataset_file['ep_len'][:].astype(np.int64)

        if observed.shape != expected.shape:
            raise RuntimeError(
                f'{condition} dataset does not match audit manifest. '
                f'hdf5_shape={observed.shape}, audit_shape={expected.shape}'
            )
        mismatch = np.flatnonzero(observed != expected)
        if len(mismatch):
            episode = int(mismatch[0])
            raise RuntimeError(
                f'{condition} dataset does not match audit manifest. '
                f'first mismatch episode={episode}, '
                f'hdf5={int(observed[episode])}, '
                f'audit={int(expected[episode])}'
            )


def validate_audit(
    audit: list[dict[str, Any]],
    num_scenarios: int,
) -> None:
    """Require both legacy experts to solve every replayed scenario."""
    _validate_audit_rows(audit, num_scenarios, require_success=True)
    visible = [row for row in audit if row['condition'] == 'paired_color']
    if not all(row['teleport_count'] == 1 for row in visible):
        raise RuntimeError(
            'Visible routing did not use exactly one tunnel traversal.'
        )


def validate_audit_row_counts(
    audit: list[dict[str, Any]],
    num_scenarios: int,
) -> None:
    """Require one audit row per scenario and condition."""
    _validate_audit_rows(audit, num_scenarios, require_success=False)


# ---------------------------------------------------------------------------
# Videos and public workflow
# ---------------------------------------------------------------------------


def record_videos(
    output_dir: Path,
    scenarios: list[dict[str, Any]],
    cfg: DictConfig,
) -> None:
    """Record expert and random-control videos for one scenario."""
    video_dir = output_dir / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    scenario = scenarios[0]

    for condition in CONDITIONS:
        options = {'scenario': {**scenario, 'identity_mode': condition}}
        policies = (
            ('policy', build_expert_policy(condition, cfg.policy_seed, cfg)),
            ('random', RandomPolicy(seed=cfg.policy_seed)),
        )
        for label, policy in policies:
            target = video_dir / f'{condition}_{label}'
            target.mkdir(parents=True, exist_ok=True)
            world = _world_for(condition, cfg)
            try:
                world.set_policy(policy)
                world.evaluate(
                    episodes=1,
                    seed=cfg.scenario_seed,
                    options=options,
                    video=target,
                    reset_mode='wait',
                )
            finally:
                world.close()


def collect_full_datasets(cfg: DictConfig) -> dict[str, Any]:
    """Collect the configured corpus and return its audit manifest."""
    collection_mode = str(cfg.get('collection_mode', 'legacy'))
    if collection_mode == 'manifest_first':
        from scripts.data.collect_tunnel_rooms_paired import (
            collect_manifest_first_datasets,
        )

        return collect_manifest_first_datasets(cfg)
    if collection_mode != 'legacy':
        raise ValueError(
            f'Unknown Tunnel Rooms collection_mode: {collection_mode!r}.'
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = dataset_names_from_cfg(cfg)
    paths = {
        condition: str(
            record_condition(
                output_dir,
                condition,
                cfg,
                dataset_name=dataset_names[condition],
                cache_dir=Path(cfg.dataset_cache_dir),
            )
        )
        for condition in CONDITIONS
    }

    if bool(cfg.get('require_all_raw_success', True)):
        audit, video_scenario = audit_seeded_scenarios(cfg)
        validate_audit(audit, cfg.num_scenarios)
        validate_datasets_match_audit(paths, audit)
    else:
        audit, video_scenario = audit_from_stored_datasets(cfg, paths)
        validate_audit_row_counts(audit, cfg.num_scenarios)

    manifest = {
        'environment': 'swm/TunnelRooms-v1',
        'num_leaves': int(cfg.get('num_leaves', 2)),
        'conditions': list(CONDITIONS),
        'dataset_names': dataset_names,
        'training_keys': TRAINING_KEYS,
        'num_paired_scenarios': cfg.num_scenarios,
        'datasets': paths,
        'episodes': audit,
        'scenario_constraints': dict(cfg.get('scenario_constraints') or {}),
        'collection_policy': {
            'hidden': 'TunnelSearchPolicy',
            'paired_color': 'VisibleRoutingPolicy',
            **policy_settings(cfg),
        },
        'prototype_caveat': (
            'Natural endpoint landing may reveal route information from the '
            'current position; interpret route-rank results accordingly.'
        ),
    }
    with (output_dir / 'audit_manifest.json').open('w') as file:
        json.dump(manifest, file, indent=2)

    if cfg.record_videos:
        record_videos(output_dir, [video_scenario], cfg)
    return manifest


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _world_for(condition: str, cfg: DictConfig) -> Any:
    """Create the common rendered world used by collection and videos."""
    return swm.World(
        'swm/TunnelRooms-v1',
        **cfg.world,
        render_mode='rgb_array',
        identity_mode=condition,
        num_leaves=int(cfg.get('num_leaves', 2)),
    )


def _write_dataset(
    dataset_file: h5py.File,
    name: str,
    values: np.ndarray,
    **create_kwargs: Any,
) -> None:
    """Replace a derived row-metadata column only when its shape changed."""
    if name in dataset_file and dataset_file[name].shape == values.shape:
        dataset_file[name][:] = values
        return
    if name in dataset_file:
        del dataset_file[name]
    dataset_file.create_dataset(name, data=values, **create_kwargs)


def _dataset_episode_status(path: str) -> dict[str, np.ndarray]:
    """Read episode lengths and terminal state from one HDF5 corpus."""
    with h5py.File(path, 'r') as dataset_file:
        lengths = dataset_file['ep_len'][:].astype(np.int64)
        offsets = dataset_file['ep_offset'][:].astype(np.int64)
        terminal_rows = offsets + lengths - 1
        terminated = dataset_file['terminated'][terminal_rows].astype(bool)
        truncated = dataset_file['truncated'][terminal_rows].astype(bool)
    return {
        'lengths': lengths,
        'terminated': terminated,
        'truncated': truncated,
    }


def _validate_audit_rows(
    audit: list[dict[str, Any]],
    num_scenarios: int,
    *,
    require_success: bool,
) -> None:
    """Enforce the row-count and optional success contract for each condition."""
    for condition in CONDITIONS:
        rows = [row for row in audit if row['condition'] == condition]
        if len(rows) != num_scenarios:
            raise RuntimeError(f'Incomplete audit rows for {condition}.')
        if require_success and not all(row['success'] for row in rows):
            raise RuntimeError(f'{condition} policy failed a pilot scenario.')


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='tunnel_rooms',
)
def run(cfg: DictConfig) -> None:
    """Collect Tunnel Rooms datasets from the selected Hydra configuration."""
    collect_full_datasets(cfg)
    build_stratified_eval_manifest(cfg, Path(cfg.output_dir))
    logging.success(
        f'Tunnel Rooms datasets written to '
        f'{Path(cfg.dataset_cache_dir) / "datasets"}'
    )
    logging.success(f'Audit outputs written to {cfg.output_dir}')


if __name__ == '__main__':
    run()
