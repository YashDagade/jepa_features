"""Collect, audit, and preview selected Chaser benchmark conditions.

The calibration command is intentionally separate from full publication.  It
must pass the TR2-Color exposure tolerances and be reviewed before switching
``task`` to ``collect`` for the 10,000-episode datasets.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.data import HDF5Writer
from stable_worldmodel.envs.chaser import ChaserExpertPolicy
from stable_worldmodel.envs.chaser.env import CUE_COLORS, CUE_NAMES
from stable_worldmodel.plot import save_video


CONDITIONS = ('reflection', 'cue')
TRAINING_KEYS = ['pixels', 'action']
TR2_COLOR_TARGETS = {
    'raw_event_rate': 0.018699422897120933,
    'predicted_interval_exposure': 0.0895,
    'event_clip_exposure': 0.2662,
}
CALIBRATION_TOLERANCES = {
    'raw_event_rate': 0.001,
    'predicted_interval_exposure': 0.005,
    'event_clip_exposure': 0.01,
}


def loader_exposure(
    episode_events: list[np.ndarray],
    *,
    frameskip: int = 5,
    num_observations: int = 4,
) -> dict[str, float | int]:
    """Measure event exposure under the actual Stable WM clip indexer.

    An event stored at row ``j`` belongs to the transition ending at row
    ``j``.  For observations at ``s, s+f, ...``, prediction interval ``k``
    therefore reads event rows ``s+k*f+1`` through ``s+(k+1)*f``.
    """
    interval_hits = 0
    interval_total = 0
    event_clips = 0
    legal_clips = 0
    span = frameskip * num_observations

    # 1. Reproduce Dataset.clip_indices episode by episode.
    for events in episode_events:
        events = np.asarray(events, dtype=bool).reshape(-1)
        starts = max(0, len(events) - span + 1)
        if not starts:
            continue
        prefix = np.concatenate(([0], np.cumsum(events, dtype=np.int64)))

        # 2. Count each of the three predicted intervals and the union per clip.
        for start in range(starts):
            clip_has_event = False
            for interval in range(num_observations - 1):
                lower = start + interval * frameskip + 1
                upper = start + (interval + 1) * frameskip + 1
                hit = bool(prefix[upper] - prefix[lower])
                interval_hits += int(hit)
                interval_total += 1
                clip_has_event |= hit
            event_clips += int(clip_has_event)
        legal_clips += starts

    return {
        'legal_clips': legal_clips,
        'predicted_intervals': interval_total,
        'event_intervals': interval_hits,
        'event_clips': event_clips,
        'predicted_interval_exposure': interval_hits / interval_total,
        'event_clip_exposure': event_clips / legal_clips,
    }


def simulate_condition(
    condition: str,
    *,
    episodes: int,
    episode_steps: int,
    seed: int,
    action_noise: float,
    runner_noise_std: float,
    cue_ids: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Run uncompressed episodes and return the calibration audit."""
    _validate_condition(condition)
    active_cues = _active_cue_ids(cue_ids)
    cue_index = {cue_id: index for index, cue_id in enumerate(active_cues)}
    env = gym.make(
        'swm/Chaser-v0',
        mapping_mode=condition,
        max_episode_steps=episode_steps,
        runner_noise_std=runner_noise_std,
        cue_ids=active_cues,
    )
    policy = ChaserExpertPolicy(action_noise=action_noise, seed=seed)
    policy.set_env(env)
    event_sequences: list[np.ndarray] = []
    cue_quadrant = np.zeros((len(active_cues), 4), dtype=np.int64)
    cue_events = np.zeros(len(active_cues), dtype=np.int64)
    cue_episodes = np.zeros(len(active_cues), dtype=np.int64)
    directions = {-1: 0, 1: 0}
    immediate_retriggers = 0
    boundary_contacts = 0

    try:
        # 1. Simulate complete Gymnasium episodes using the collection policy.
        for episode in range(episodes):
            _, info = env.reset(seed=seed + episode)
            directions[int(info['orbit_direction'])] += 1
            if condition == 'cue':
                cue_episodes[cue_index[int(info['cue_id'])]] += 1
            events: list[bool] = []
            previous_event = False
            first_contact_recorded = False
            while True:
                _, _, terminated, truncated, info = env.step(
                    policy.get_action(info)
                )
                event = bool(info['event'])
                blue = np.asarray(info['pos_blue'])
                boundary_contacts += int(
                    np.any(blue <= 25.0) or np.any(blue >= 199.0)
                )
                immediate_retriggers += int(event and previous_event)
                previous_event = event
                events.append(event)
                if event and condition == 'cue':
                    cue_id = int(info['cue_id'])
                    cue_slot = cue_index[cue_id]
                    cue_events[cue_slot] += 1
                    # Sampling balance is defined by the first contact in each
                    # episode. Later contacts are causal outcomes of the first
                    # cue-conditioned teleport and must not bias this audit.
                    if not first_contact_recorded:
                        cue_quadrant[
                            cue_slot, int(info['contact_quadrant'])
                        ] += 1
                        first_contact_recorded = True
                if terminated or truncated:
                    break
            event_sequences.append(np.asarray(events, dtype=bool))
    finally:
        env.close()

    # 2. Measure raw and loader-realized rarity from the same event sequences.
    exposure = loader_exposure(event_sequences)
    total_steps = sum(map(len, event_sequences))
    total_events = sum(int(events.sum()) for events in event_sequences)
    audit = {
        'condition': condition,
        'episodes': episodes,
        'raw_frames': total_steps,
        'events': total_events,
        'events_per_episode': total_events / episodes,
        'raw_event_rate': total_events / total_steps,
        **exposure,
        'clockwise_episodes': directions[-1],
        'counterclockwise_episodes': directions[1],
        'immediate_retriggers': immediate_retriggers,
        'boundary_contacts': boundary_contacts,
        'cue_ids': list(active_cues),
        'cue_names': [CUE_NAMES[cue_id] for cue_id in active_cues],
        'cue_quadrant_counts': cue_quadrant.tolist(),
        'cue_event_counts': cue_events.tolist(),
        'cue_episode_counts': cue_episodes.tolist(),
    }
    audit['calibration'] = calibration_checks(audit)
    return audit, event_sequences


def calibration_checks(audit: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the quantitative calibration gate without hiding failures."""
    checks: dict[str, Any] = {}
    for metric, target in TR2_COLOR_TARGETS.items():
        difference = abs(float(audit[metric]) - target)
        checks[metric] = {
            'target': target,
            'observed': float(audit[metric]),
            'tolerance': CALIBRATION_TOLERANCES[metric],
            'passed': difference <= CALIBRATION_TOLERANCES[metric],
        }
    checks['no_immediate_retrigger'] = audit['immediate_retriggers'] == 0
    checks['no_boundary_contact'] = audit['boundary_contacts'] == 0
    direction_mean = audit['episodes'] / 2
    checks['orbit_direction_balance'] = bool(
        abs(audit['clockwise_episodes'] - direction_mean)
        <= 0.1 * direction_mean
    )
    if audit['condition'] == 'cue':
        counts = np.asarray(audit['cue_quadrant_counts'], dtype=np.float64)
        mean = counts.mean()
        checks['cue_quadrant_balance'] = bool(
            mean > 0 and np.max(np.abs(counts - mean)) <= 0.1 * mean
        )
        episode_counts = np.asarray(
            audit['cue_episode_counts'], dtype=np.float64
        )
        checks['cue_episode_balance'] = bool(
            np.max(np.abs(episode_counts - episode_counts.mean()))
            <= 0.1 * episode_counts.mean()
        )
    return checks


def record_previews(cfg: DictConfig, output_dir: Path) -> list[dict[str, Any]]:
    """Save reflection controls and one clip for every target-color cue."""
    records = []
    active_cues = _active_cue_ids(cfg.world.get('cue_ids'))
    preview_specs = [
        ('reflection', -1, -1, 'reflection_cw'),
        ('reflection', 1, -1, 'reflection_ccw'),
        *[
            ('cue', 1, cue_id, f'cue_{cue_name}')
            for cue_id, cue_name in (
                (cue_id, CUE_NAMES[cue_id]) for cue_id in active_cues
            )
        ],
    ]
    for condition, direction, cue_id, name in preview_specs:
        env = gym.make(
            'swm/Chaser-v0',
            mapping_mode=condition,
            max_episode_steps=int(cfg.world.max_episode_steps),
            runner_noise_std=float(cfg.world.runner_noise_std),
            cue_ids=active_cues,
        )
        policy = ChaserExpertPolicy(
            action_noise=float(cfg.policy.action_noise), seed=int(cfg.seed)
        )
        policy.set_env(env)
        frames = []
        try:
            _, info = env.reset(
                seed=int(cfg.seed),
                options={
                    'scenario': {
                        'scenario_seed': int(cfg.seed),
                        'orbit_direction': direction,
                        'cue_id': cue_id,
                    }
                },
            )
            frames.append(env.render())
            while True:
                _, _, terminated, truncated, info = env.step(
                    policy.get_action(info)
                )
                frames.append(env.render())
                if terminated or truncated:
                    break
            path = output_dir / f'{name}.mp4'
            save_video(path, frames)
            records.append(
                {
                    'condition': condition,
                    'cue': CUE_NAMES[cue_id] if cue_id >= 0 else None,
                    'orbit_direction': direction,
                    'path': str(path),
                    'frames': len(frames),
                    'events': int(info['event_count']),
                }
            )
        finally:
            env.close()
    return records


def collect_condition(
    condition: str, cfg: DictConfig, dataset_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Transactionally collect and validate one HDF5 condition."""
    _validate_condition(condition)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename = cfg.get('dataset_filename') or f'chaser_{condition}.h5'
    if Path(filename).name != str(filename):
        raise ValueError('dataset_filename must be a plain filename.')
    destination = dataset_dir / str(filename)
    partial = destination.with_suffix('.partial.h5')
    if destination.exists() and not cfg.overwrite:
        raise FileExistsError(
            f'{destination} already exists; set overwrite=true explicitly.'
        )
    if partial.exists():
        raise FileExistsError(
            f'Incomplete collection exists at {partial}; inspect or remove it.'
        )

    # 1. Reuse the maintained World collection loop and generic HDF5 writer.
    world = swm.World(
        'swm/Chaser-v0',
        num_envs=int(cfg.world.num_envs),
        image_shape=tuple(cfg.world.image_shape),
        max_episode_steps=int(cfg.world.max_episode_steps),
        mapping_mode=condition,
        runner_noise_std=float(cfg.world.runner_noise_std),
        cue_ids=_active_cue_ids(cfg.world.get('cue_ids')),
    )
    world.set_policy(
        ChaserExpertPolicy(
            action_noise=float(cfg.policy.action_noise), seed=int(cfg.seed)
        )
    )
    try:
        world.collect(
            writer=HDF5Writer(partial, mode='error'),
            episodes=int(cfg.episodes),
            seed=int(cfg.seed),
        )
    finally:
        world.close()

    # 2. Audit the completed temporary file before publishing it atomically.
    audit = audit_hdf5(partial)
    if condition == 'cue':
        expected_cues = list(_active_cue_ids(cfg.world.get('cue_ids')))
        if audit['cue_ids'] != expected_cues:
            raise ValueError(
                'Collected cue IDs do not match world.cue_ids: '
                f'expected {expected_cues}, found {audit["cue_ids"]}.'
            )
    if destination.exists():
        destination.unlink()
    os.replace(partial, destination)
    audit['sha256'] = _sha256(destination)
    audit['path'] = str(destination)
    return destination, audit


def audit_hdf5(path: Path) -> dict[str, Any]:
    """Build the dataset and loader-exposure portion of a corpus manifest."""
    with h5py.File(path, 'r') as data:
        lengths = data['ep_len'][:].astype(np.int64)
        offsets = data['ep_offset'][:].astype(np.int64)
        event_sequences = [
            data['event'][offset : offset + length].astype(bool)
            for offset, length in zip(offsets, lengths)
        ]
        events = data['event'][:].astype(bool)
        cue = data['cue_id'][:].astype(np.int64)
        active_cues = tuple(
            int(cue_id) for cue_id in np.unique(cue) if cue_id >= 0
        )
        cue_index = {cue_id: index for index, cue_id in enumerate(active_cues)}
        quadrant = data['contact_quadrant'][:].astype(np.int64)
        event_rows = np.flatnonzero(events)
        cue_quadrant = np.zeros((len(active_cues), 4), dtype=np.int64)
        cue_events = np.zeros(len(active_cues), dtype=np.int64)
        cue_episodes = np.zeros(len(active_cues), dtype=np.int64)
        for offset, length in zip(offsets, lengths):
            if cue[offset] >= 0:
                cue_episodes[cue_index[int(cue[offset])]] += 1
                episode_events = np.flatnonzero(
                    events[offset : offset + length]
                )
                if len(episode_events):
                    first = offset + episode_events[0]
                    cue_quadrant[
                        cue_index[int(cue[first])], quadrant[first]
                    ] += 1
        for cue_id in cue[event_rows]:
            if cue_id >= 0:
                cue_events[cue_index[int(cue_id)]] += 1
        intervals = [
            int(right - left)
            for sequence in event_sequences
            for left, right in zip(
                np.flatnonzero(sequence)[:-1], np.flatnonzero(sequence)[1:]
            )
        ]
        columns = sorted(
            key for key in data.keys() if key not in {'ep_len', 'ep_offset'}
        )

    exposure = loader_exposure(event_sequences)
    return {
        'episodes': len(lengths),
        'raw_frames': int(lengths.sum()),
        'events': int(events.sum()),
        'events_per_episode': float(events.sum() / len(lengths)),
        'raw_event_rate': float(events.mean()),
        **exposure,
        'episode_length_min': int(lengths.min()),
        'episode_length_max': int(lengths.max()),
        'cue_ids': list(active_cues),
        'cue_names': [CUE_NAMES[cue_id] for cue_id in active_cues],
        'cue_quadrant_counts': cue_quadrant.tolist(),
        'cue_event_counts': cue_events.tolist(),
        'cue_episode_counts': cue_episodes.tolist(),
        'inter_event_interval_mean': (
            float(np.mean(intervals)) if intervals else None
        ),
        'columns': columns,
    }


def run_calibration(cfg: DictConfig) -> dict[str, Any]:
    """Produce the reviewable pre-collection audit and representative clips."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audits = {}
    for condition in _selected_conditions(cfg):
        audit, _ = simulate_condition(
            condition,
            episodes=int(cfg.calibration.episodes),
            episode_steps=int(cfg.world.max_episode_steps),
            seed=int(cfg.seed),
            action_noise=float(cfg.policy.action_noise),
            runner_noise_std=float(cfg.world.runner_noise_std),
            cue_ids=cfg.world.get('cue_ids'),
        )
        audits[condition] = audit
    result = {
        'task': 'calibrate',
        'git_commit': _git_commit(),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'tr2_color_targets': TR2_COLOR_TARGETS,
        'conditions': audits,
        'previews': record_previews(cfg, output_dir),
    }
    _write_json(output_dir / 'calibration_audit.json', result)
    return result


def run_collection(cfg: DictConfig) -> dict[str, Any]:
    """Collect the reviewed conditions and publish one canonical manifest."""
    if not cfg.calibration_approved:
        raise RuntimeError(
            'Full Chaser collection requires calibration_approved=true after '
            'the preview clips and calibration_audit.json have been reviewed.'
        )
    dataset_dir = Path(cfg.dataset_cache_dir) / 'datasets'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    selected_conditions = _selected_conditions(cfg)
    if cfg.get('dataset_filename') and len(selected_conditions) != 1:
        raise ValueError(
            'dataset_filename requires exactly one selected condition.'
        )
    datasets = {}
    for condition in selected_conditions:
        _, datasets[condition] = collect_condition(condition, cfg, dataset_dir)
    manifest = {
        'schema_version': 1,
        'git_commit': _git_commit(),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'training_keys': TRAINING_KEYS,
        'loader_contract': {
            'frameskip': 5,
            'num_observations': 4,
            'predicted_intervals': 3,
        },
        'datasets': datasets,
    }
    output = Path(cfg.output_dir) / 'corpus_manifest.json'
    _write_json(output, manifest)
    return manifest


@hydra.main(version_base=None, config_path='./config', config_name='chaser')
def run(cfg: DictConfig) -> None:
    if cfg.task == 'calibrate':
        result = run_calibration(cfg)
    elif cfg.task == 'collect':
        result = run_collection(cfg)
    else:
        raise ValueError('task must be exactly "calibrate" or "collect".')
    print(json.dumps(result, indent=2, sort_keys=True))


def _validate_condition(condition: str) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f'condition must be one of {CONDITIONS}.')


def _selected_conditions(cfg: DictConfig) -> tuple[str, ...]:
    """Return a non-empty, duplicate-free list of requested conditions."""
    selected = tuple(str(condition) for condition in cfg.conditions)
    if not selected:
        raise ValueError(
            'conditions must select at least one Chaser condition.'
        )
    if len(selected) != len(set(selected)):
        raise ValueError('conditions must not contain duplicates.')
    for condition in selected:
        _validate_condition(condition)
    return selected


def _active_cue_ids(values: Sequence[int] | None) -> tuple[int, ...]:
    """Validate the ordered subset of persistent runner-color cues."""
    active = (
        tuple(range(len(CUE_COLORS)))
        if values is None
        else tuple(int(value) for value in values)
    )
    if not active or len(set(active)) != len(active):
        raise ValueError('world.cue_ids must be non-empty and duplicate-free.')
    if not set(active) <= set(range(len(CUE_COLORS))):
        raise ValueError(
            f'world.cue_ids must be selected from {list(range(len(CUE_COLORS)))}.'
        )
    return active


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True
    ).strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


if __name__ == '__main__':
    run()
