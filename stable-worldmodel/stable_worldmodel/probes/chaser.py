"""Episode-disjoint frozen linear probes for the Chaser-Cue benchmark.

The probe separates the common smooth-motion objective from the rare
cue-conditioned teleport objective. Privileged labels are read only during
evaluation; only pixels and actions remain the model inputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stable_worldmodel.probes.location import (
    FrozenLocationFeatures,
    aligned_history_inputs,
    assert_episode_disjoint,
    fit_ridge_classifier,
    fit_standardized_ridge,
    frozen_inference,
    hdf5_column_stats,
    regression_metrics,
)

FEATURE_SOURCES = ('encoder', 'projector', 'predictor')
SPLIT_NAMES = ('fit', 'validation', 'test')
SPLIT_TO_CODE = {name: code for code, name in enumerate(SPLIT_NAMES)}


def _hash_arrays(named_arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    """Hash shape, dtype, name, and bytes for deterministic sample provenance."""
    hasher = hashlib.sha256()
    for name, value in named_arrays:
        array = np.ascontiguousarray(value)
        hasher.update(name.encode())
        hasher.update(array.dtype.str.encode())
        hasher.update(np.asarray(array.shape, np.int64).tobytes())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _choose(
    rng: np.random.Generator, values: np.ndarray, limit: int
) -> np.ndarray:
    """Select a deterministic subset without replacement."""
    values = np.unique(np.asarray(values, dtype=np.int64))
    if len(values) <= limit:
        return values
    return np.sort(rng.choice(values, size=limit, replace=False))


def _episode_splits(cues: np.ndarray, seed: int) -> np.ndarray:
    """Stratify whole episodes by cue before any frame is selected."""
    rng = np.random.default_rng(seed)
    splits = np.full(len(cues), -1, dtype=np.int8)

    # 1. Split each cue independently so every active class remains balanced.
    for cue in np.unique(cues):
        indices = np.flatnonzero(cues == cue)
        rng.shuffle(indices)
        fit_end = int(0.70 * len(indices))
        validation_end = fit_end + int(0.15 * len(indices))
        splits[indices[:fit_end]] = SPLIT_TO_CODE['fit']
        splits[indices[fit_end:validation_end]] = SPLIT_TO_CODE['validation']
        splits[indices[validation_end:]] = SPLIT_TO_CODE['test']
    if np.any(splits < 0):
        raise RuntimeError('Every Chaser episode must receive one split.')
    return splits


@dataclass(frozen=True)
class ChaserProbeSamples:
    """Immutable sample rows, task masks, and privileged evaluation labels."""

    source_path: str
    frame_rows: np.ndarray
    future_rows: np.ndarray
    episode_ids: np.ndarray
    split_codes: np.ndarray
    general_mask: np.ndarray
    event_balanced_mask: np.ndarray
    first_contact_mask: np.ndarray
    event_imminent: np.ndarray
    cue_id: np.ndarray
    cue_values: np.ndarray
    contact_quadrant: np.ndarray
    destination_quadrant: np.ndarray
    red_current: np.ndarray
    red_future: np.ndarray
    runner_current: np.ndarray
    runner_future: np.ndarray
    frameskip: int
    history_size: int
    sample_sha256: str

    @property
    def sample_ids(self) -> np.ndarray:
        """Use the unique current-frame row as the sample identifier."""
        return self.frame_rows[:, -1]

    def summary(self) -> dict[str, Any]:
        """Return compact counts needed to interpret every result table."""
        counts = {}
        for code, split in enumerate(SPLIT_NAMES):
            split_mask = self.split_codes == code
            counts[split] = {
                'samples': int(split_mask.sum()),
                'episodes': int(np.unique(self.episode_ids[split_mask]).size),
                'general': int((split_mask & self.general_mask).sum()),
                'event_balanced': int(
                    (split_mask & self.event_balanced_mask).sum()
                ),
                'first_contact': int(
                    (split_mask & self.first_contact_mask).sum()
                ),
            }
        return {
            'counts': counts,
            'cue_values': self.cue_values.tolist(),
            'frameskip': self.frameskip,
            'history_size': self.history_size,
            'sample_sha256': self.sample_sha256,
        }


def build_chaser_probe_samples(
    source_path: str | Path,
    *,
    seed: int = 3072,
    frameskip: int = 5,
    history_size: int = 3,
    samples_per_split: int = 2048,
) -> ChaserProbeSamples:
    """Select ordinary, balanced-event, and first-contact probe histories."""
    source_path = Path(source_path).resolve()
    if frameskip <= 0 or history_size <= 0 or samples_per_split <= 0:
        raise ValueError(
            'frameskip, history_size, and samples_per_split must be positive.'
        )
    with h5py.File(source_path, 'r') as source:
        required = {
            'action',
            'contact_quadrant',
            'cue_id',
            'destination_quadrant',
            'ep_len',
            'ep_offset',
            'event',
            'id',
            'pixels',
            'pos_blue',
            'pos_red',
        }
        missing = sorted(required - set(source))
        if missing:
            raise ValueError(f'Chaser HDF5 is missing columns: {missing}.')

        offsets = source['ep_offset'][:].astype(np.int64)
        lengths = source['ep_len'][:].astype(np.int64)
        events = source['event'][:].astype(bool)
        cue_by_episode = source['cue_id'][offsets].astype(np.int8)
        cue_values = np.unique(cue_by_episode)
        if len(cue_values) < 2 or np.any(cue_values < 0):
            raise ValueError(
                'Chaser cue probes require at least two non-negative cues.'
            )
        episode_splits = _episode_splits(cue_by_episode, seed)
        rng = np.random.default_rng(seed)

        # 1. Enumerate every legal current frame and label whether its next
        # frameskip interval contains a contact.
        legal = [[] for _ in SPLIT_NAMES]
        event_positive = [[] for _ in SPLIT_NAMES]
        event_negative = [[] for _ in SPLIT_NAMES]
        first_contacts = [[] for _ in SPLIT_NAMES]
        for episode, (offset, length) in enumerate(
            zip(offsets, lengths, strict=True)
        ):
            split = int(episode_splits[episode])
            first_current = offset + (history_size - 1) * frameskip
            final_current = offset + length - frameskip - 1
            if first_current > final_current:
                continue
            current = np.arange(first_current, final_current + 1)
            episode_events = events[offset : offset + length]
            prefix = np.concatenate(
                ([0], np.cumsum(episode_events, dtype=np.int64))
            )
            relative = current - offset
            imminent = (
                prefix[relative + frameskip + 1] - prefix[relative + 1] > 0
            )
            legal[split].append(current)
            event_positive[split].append(current[imminent])
            event_negative[split].append(current[~imminent])

            # The first contact is the clean causal test: no earlier
            # cue-conditioned teleport can have altered this trajectory.
            event_rows = np.flatnonzero(episode_events)
            if len(event_rows):
                event_row = offset + int(event_rows[0])
                pre_event = event_row - frameskip
                if pre_event >= first_current:
                    first_contacts[split].append(
                        np.asarray([pre_event], dtype=np.int64)
                    )

        def merged(values: list[np.ndarray]) -> np.ndarray:
            return (
                np.concatenate(values)
                if values
                else np.empty(0, dtype=np.int64)
            )

        # 2. Select equal positive/negative event samples and an independent
        # ordinary-frame sample inside each episode-disjoint split.
        records: dict[int, set[str]] = {}
        for split in range(len(SPLIT_NAMES)):
            general = _choose(rng, merged(legal[split]), samples_per_split)
            positives = merged(event_positive[split])
            negatives = merged(event_negative[split])
            balanced_count = min(
                samples_per_split,
                len(np.unique(positives)),
                len(np.unique(negatives)),
            )
            positives = _choose(rng, positives, balanced_count)
            negatives = _choose(rng, negatives, balanced_count)
            contacts = _choose(
                rng, merged(first_contacts[split]), samples_per_split
            )
            for name, rows in (
                ('general', general),
                ('event_balanced', positives),
                ('event_balanced', negatives),
                ('first_contact', contacts),
            ):
                for row in rows:
                    records.setdefault(int(row), set()).add(name)

        # 3. Materialize histories and labels in one deterministic row order.
        current_rows = np.asarray(sorted(records), dtype=np.int64)
        frame_rows = current_rows[:, None] - frameskip * np.arange(
            history_size - 1, -1, -1, dtype=np.int64
        )
        future_rows = current_rows + frameskip
        row_episode = np.searchsorted(offsets, current_rows, side='right') - 1
        split_codes = episode_splits[row_episode]
        episode_ids = source['id'][current_rows].astype(np.int64)
        general_mask = np.asarray(
            ['general' in records[int(row)] for row in current_rows], bool
        )
        event_balanced_mask = np.asarray(
            ['event_balanced' in records[int(row)] for row in current_rows],
            bool,
        )
        first_contact_mask = np.asarray(
            ['first_contact' in records[int(row)] for row in current_rows],
            bool,
        )
        event_imminent = np.asarray(
            [
                events[row + 1 : row + frameskip + 1].any()
                for row in current_rows
            ],
            bool,
        )
        cue_id = source['cue_id'][current_rows].astype(np.int8)
        contact_quadrant = np.full(len(current_rows), -1, dtype=np.int8)
        destination_quadrant = np.full(len(current_rows), -1, dtype=np.int8)
        contact_quadrant[first_contact_mask] = source['contact_quadrant'][
            future_rows[first_contact_mask]
        ]
        destination_quadrant[first_contact_mask] = source[
            'destination_quadrant'
        ][future_rows[first_contact_mask]]
        red_current = source['pos_red'][current_rows].astype(np.float32)
        red_future = source['pos_red'][future_rows].astype(np.float32)
        runner_current = source['pos_blue'][current_rows].astype(np.float32)
        runner_future = source['pos_blue'][future_rows].astype(np.float32)

    # 4. Reject temporal leakage, misaligned first contacts, and class drift.
    assert_episode_disjoint(episode_ids, split_codes)
    if not np.all(events[future_rows[first_contact_mask]]):
        raise ValueError('First-contact samples must end exactly on an event.')
    if np.any(contact_quadrant[first_contact_mask] < 0) or np.any(
        destination_quadrant[first_contact_mask] < 0
    ):
        raise ValueError('First-contact labels are incomplete.')
    for split in range(len(SPLIT_NAMES)):
        mask = (split_codes == split) & event_balanced_mask
        if int(event_imminent[mask].sum()) * 2 != int(mask.sum()):
            raise ValueError('Event-imminence samples must be class-balanced.')

    sample_sha256 = _hash_arrays(
        (
            ('frame_rows', frame_rows),
            ('future_rows', future_rows),
            ('episode_ids', episode_ids),
            ('split_codes', split_codes),
            ('general_mask', general_mask),
            ('event_balanced_mask', event_balanced_mask),
            ('first_contact_mask', first_contact_mask),
            ('event_imminent', event_imminent),
            ('cue_id', cue_id),
            ('cue_values', cue_values),
            ('contact_quadrant', contact_quadrant),
            ('destination_quadrant', destination_quadrant),
        )
    )
    return ChaserProbeSamples(
        source_path=str(source_path),
        frame_rows=frame_rows,
        future_rows=future_rows,
        episode_ids=episode_ids,
        split_codes=split_codes,
        general_mask=general_mask,
        event_balanced_mask=event_balanced_mask,
        first_contact_mask=first_contact_mask,
        event_imminent=event_imminent,
        cue_id=cue_id,
        cue_values=cue_values,
        contact_quadrant=contact_quadrant,
        destination_quadrant=destination_quadrant,
        red_current=red_current,
        red_future=red_future,
        runner_current=runner_current,
        runner_future=runner_future,
        frameskip=frameskip,
        history_size=history_size,
        sample_sha256=sample_sha256,
    )


class ChaserProbeDataset(Dataset):
    """Read the exact visual histories selected by ``ChaserProbeSamples``."""

    def __init__(self, samples: ChaserProbeSamples) -> None:
        self.samples = samples
        self._source: h5py.File | None = None
        with h5py.File(samples.source_path, 'r') as source:
            self.action_input_mean, self.action_input_std = hdf5_column_stats(
                source['action']
            )

    def _open_source(self) -> h5py.File:
        if self._source is None:
            self._source = h5py.File(self.samples.source_path, 'r', swmr=True)
        return self._source

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state['_source'] = None
        return state

    def __len__(self) -> int:
        return len(self.samples.frame_rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = aligned_history_inputs(
            self._open_source(),
            self.samples.frame_rows[index],
            self.samples.frameskip,
            sample_id=int(self.samples.sample_ids[index]),
        )
        item['sample_index'] = torch.tensor(index, dtype=torch.int64)
        return item


def extract_chaser_features(
    dataset: ChaserProbeDataset,
    extractor: FrozenLocationFeatures,
    *,
    device: torch.device | str,
    batch_size: int = 128,
    num_workers: int = 0,
    autocast: bool = True,
) -> np.ndarray:
    """Extract encoder/projector/predictor vectors in immutable sample order."""
    device = torch.device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == 'cuda',
        generator=torch.Generator().manual_seed(0),
    )
    batches: list[torch.Tensor] = []
    extractor.to(device)
    dtype = torch.bfloat16 if autocast and device.type == 'cuda' else None

    # 1. Freeze every nested training flag and use one optional CUDA autocast.
    context = (
        torch.autocast(device_type='cuda', dtype=dtype)
        if dtype is not None
        else torch.no_grad()
    )
    with frozen_inference(extractor), context:
        for batch in loader:
            features = extractor.representation_features(
                batch['pixels'].to(device, non_blocking=True),
                batch['action_history'].to(device, non_blocking=True),
            )
            if features.ndim != 3 or features.shape[1] != len(FEATURE_SOURCES):
                raise ValueError(
                    'Chaser readout must return (batch, 3, feature), got '
                    f'{tuple(features.shape)}.'
                )
            if not torch.isfinite(features).all():
                raise FloatingPointError('Chaser features are not finite.')
            batches.append(features.float().cpu())
    return torch.cat(batches).numpy()


def _fit_positions(
    features: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    names: tuple[str, ...],
    alpha: float,
) -> dict[str, Any]:
    model = fit_standardized_ridge(
        features[train_mask], targets[train_mask], alpha
    )
    prediction = model.predict(features[test_mask])
    metrics = regression_metrics(targets[test_mask], prediction, names)
    baseline = np.broadcast_to(
        targets[train_mask].mean(axis=0), targets[test_mask].shape
    )
    metrics['mean_baseline'] = regression_metrics(
        targets[test_mask], baseline, names
    )
    return metrics


def fit_chaser_probes(
    features: np.ndarray,
    samples: ChaserProbeSamples,
    *,
    ridge_alpha: float = 10.0,
) -> dict[str, Any]:
    """Fit identical frozen linear tasks at all three representation stages."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 3 or x.shape[:2] != (
        len(samples.frame_rows),
        len(FEATURE_SOURCES),
    ):
        raise ValueError('Chaser features do not match sample/stage counts.')
    fit_or_validation = samples.split_codes != SPLIT_TO_CODE['test']
    test = samples.split_codes == SPLIT_TO_CODE['test']
    results: dict[str, Any] = {}
    if not np.all(np.isin(samples.cue_id, samples.cue_values)):
        raise ValueError('Every probe cue must belong to cue_values.')
    cue_classes = np.searchsorted(samples.cue_values, samples.cue_id)
    cue_class_count = len(samples.cue_values)

    # 1. Use current positions for encoder/projector and aligned future
    # positions for the predictor, matching what each vector represents.
    current_positions = np.concatenate(
        (samples.red_current, samples.runner_current), axis=1
    )
    future_positions = np.concatenate(
        (samples.red_future, samples.runner_future), axis=1
    )
    position_names = ('red_x', 'red_y', 'runner_x', 'runner_y')
    for stage_index, stage in enumerate(FEATURE_SOURCES):
        stage_features = x[:, stage_index]
        general_train = fit_or_validation & samples.general_mask
        general_test = test & samples.general_mask
        positions = (
            future_positions if stage == 'predictor' else current_positions
        )

        # 2. Test the dominant motion objective and persistent cue everywhere.
        stage_results = {
            'motion_position': _fit_positions(
                stage_features,
                positions,
                general_train,
                general_test,
                position_names,
                ridge_alpha,
            ),
            'cue': fit_ridge_classifier(
                stage_features,
                cue_classes,
                general_train,
                general_test,
                cue_class_count,
                ridge_alpha,
            ),
        }

        # 3. Balance common/non-common intervals for the anticipation probe.
        event_train = fit_or_validation & samples.event_balanced_mask
        event_test = test & samples.event_balanced_mask
        stage_results['event_imminence'] = fit_ridge_classifier(
            stage_features,
            samples.event_imminent.astype(np.int8),
            event_train,
            event_test,
            2,
            ridge_alpha,
        )

        # 4. Evaluate the rare mapping only at each episode's first contact,
        # before any previous cue-conditioned teleport can leak the cue.
        contact_train = fit_or_validation & samples.first_contact_mask
        contact_test = test & samples.first_contact_mask
        stage_results['cue_before_first_contact'] = fit_ridge_classifier(
            stage_features,
            cue_classes,
            contact_train,
            contact_test,
            cue_class_count,
            ridge_alpha,
        )
        stage_results['contact_quadrant'] = fit_ridge_classifier(
            stage_features,
            samples.contact_quadrant,
            contact_train,
            contact_test,
            4,
            ridge_alpha,
        )
        stage_results['destination_quadrant'] = fit_ridge_classifier(
            stage_features,
            samples.destination_quadrant,
            contact_train,
            contact_test,
            4,
            ridge_alpha,
        )
        stage_results['post_teleport_position'] = _fit_positions(
            stage_features,
            samples.red_future,
            contact_train,
            contact_test,
            ('red_x', 'red_y'),
            ridge_alpha,
        )
        results[stage] = stage_results

    return {
        'ridge_alpha': float(ridge_alpha),
        'feature_width': int(x.shape[-1]),
        'feature_sources': list(FEATURE_SOURCES),
        'samples': samples.summary(),
        'stages': results,
    }


def flatten_chaser_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Flatten paper-facing MAE and accuracy metrics for W&B histories."""
    metrics: dict[str, float] = {}
    for stage, tasks in result['stages'].items():
        prefix = f'chaser_probe_{stage}/test'
        motion = tasks['motion_position']['points']
        metrics[f'{prefix}/red_position/mae'] = motion['red']['mae']
        metrics[f'{prefix}/runner_position/mae'] = motion['runner']['mae']
        teleport = tasks['post_teleport_position']['points']['red']
        metrics[f'{prefix}/post_teleport_position/mae'] = teleport['mae']
        for task in (
            'cue',
            'event_imminence',
            'cue_before_first_contact',
            'contact_quadrant',
            'destination_quadrant',
        ):
            for metric in (
                'accuracy',
                'balanced_accuracy',
                'balanced_accuracy_ci95_low',
                'balanced_accuracy_ci95_high',
            ):
                metrics[f'{prefix}/{task}/{metric}'] = tasks[task][metric]
    return metrics


__all__ = [
    'ChaserProbeDataset',
    'ChaserProbeSamples',
    'FEATURE_SOURCES',
    'build_chaser_probe_samples',
    'extract_chaser_features',
    'fit_chaser_probes',
    'flatten_chaser_metrics',
]
