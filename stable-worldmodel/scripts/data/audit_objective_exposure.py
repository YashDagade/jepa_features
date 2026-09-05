"""Audit dominant-motion versus rare-teleport exposure in offline datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


def _episode_arrays(source: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    offsets = source['ep_offset'][:].astype(np.int64)
    lengths = source['ep_len'][:].astype(np.int64)
    if offsets.shape != lengths.shape or np.any(lengths <= 0):
        raise ValueError(
            'ep_offset and ep_len must be equal positive vectors.'
        )
    return offsets, lengths


def _teleport_rows(
    source: h5py.File,
    offsets: np.ndarray,
    lengths: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Mark destination rows whose preceding transition teleported the agent."""
    if 'event' in source:
        events = source['event'][:].astype(bool)
    else:
        positions = source['pos_agent'][:].astype(np.float32)
        events = np.zeros(len(positions), dtype=bool)
        for offset, length in zip(offsets, lengths, strict=True):
            episode = positions[offset : offset + length]
            events[offset + 1 : offset + length] = (
                np.linalg.norm(np.diff(episode, axis=0), axis=1) > threshold
            )
    events[offsets] = False
    return events


def classify_transitions(
    source: h5py.File,
    *,
    environment: str,
    teleport_threshold: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return approach, teleport, and other masks over destination rows."""
    offsets, lengths = _episode_arrays(source)
    teleport = _teleport_rows(
        source, offsets, lengths, threshold=float(teleport_threshold)
    )
    dominant = np.zeros(len(teleport), dtype=bool)
    valid = np.zeros(len(teleport), dtype=bool)
    positions = source['pos_agent'][:].astype(np.float32)

    # 1. Each non-initial row represents the transition that ends at that row.
    for offset, length in zip(offsets, lengths, strict=True):
        valid[offset + 1 : offset + length] = True
        episode_positions = positions[offset : offset + length]
        episode_events = np.flatnonzero(teleport[offset : offset + length])

        if environment != 'tunnel':
            raise ValueError("environment must be 'tunnel'.")

        # 2. For Tunnel Rooms, the next entered portal is the privileged target
        # of each approach segment. Its final pre-teleport position estimates
        # the portal center without requiring scenario metadata in the HDF5.
        segment_start = 0
        for event_row in episode_events:
            portal = episode_positions[event_row - 1]
            destination_rows = np.arange(segment_start + 1, event_row)
            if len(destination_rows):
                before = np.linalg.norm(
                    episode_positions[destination_rows - 1] - portal, axis=1
                )
                after = np.linalg.norm(
                    episode_positions[destination_rows] - portal, axis=1
                )
                dominant[offset + destination_rows] = after < before
            segment_start = int(event_row)

    other = valid & ~teleport & ~dominant
    return dominant, teleport, other, offsets, lengths


def loader_window_exposure(
    masks: dict[str, np.ndarray],
    offsets: np.ndarray,
    lengths: np.ndarray,
    *,
    frameskip: int,
    num_steps: int,
) -> dict[str, Any]:
    """Match the dataset loader's strided clip and prediction-interval contract."""
    if frameskip <= 0 or num_steps < 2:
        raise ValueError(
            'frameskip must be positive and num_steps at least two.'
        )
    span = frameskip * num_steps
    windows = 0
    intervals = 0
    clips_with: dict[str, int] = {name: 0 for name in masks}
    intervals_with: dict[str, int] = {name: 0 for name in masks}

    # 1. Dataset.clip_indices permits every start whose full span fits.
    for offset, length in zip(offsets, lengths, strict=True):
        starts = np.arange(max(0, length - span + 1), dtype=np.int64)
        windows += len(starts)
        intervals += len(starts) * (num_steps - 1)
        for name, mask in masks.items():
            clip_hit = np.zeros(len(starts), dtype=bool)
            episode = mask[offset : offset + length].astype(np.int64)
            prefix = np.concatenate(([0], np.cumsum(episode)))
            for step in range(num_steps - 1):
                # Prefix sums make each raw interval query constant-time.
                relative = starts + step * frameskip
                hit = prefix[relative + frameskip + 1] > prefix[relative + 1]
                intervals_with[name] += int(hit.sum())
                clip_hit |= hit
            clips_with[name] += int(clip_hit.sum())

    # 2. Report both per-prediction-interval and any-interval-in-clip exposure.
    return {
        'legal_clips': int(windows),
        'predicted_intervals': int(intervals),
        **{
            f'{name}_predicted_interval_exposure': (
                intervals_with[name] / intervals if intervals else 0.0
            )
            for name in masks
        },
        **{
            f'{name}_clip_exposure': (
                clips_with[name] / windows if windows else 0.0
            )
            for name in masks
        },
    }


def audit_dataset(
    path: str | Path,
    *,
    environment: str,
    frameskip: int = 5,
    num_steps: int = 4,
    teleport_threshold: float = 30.0,
) -> dict[str, Any]:
    """Return a JSON-serializable objective and loader-exposure audit."""
    path = Path(path).expanduser().resolve()
    with h5py.File(path, 'r') as source:
        dominant, teleport, other, offsets, lengths = classify_transitions(
            source,
            environment=environment,
            teleport_threshold=teleport_threshold,
        )
        label = 'approach'
        masks = {label: dominant, 'teleport': teleport, 'other': other}
        counts = {name: int(mask.sum()) for name, mask in masks.items()}
        transitions = int(sum(counts.values()))
        return {
            'dataset': str(path),
            'environment': environment,
            'episodes': int(len(lengths)),
            'frames': int(lengths.sum()),
            'raw_transitions': transitions,
            'raw_counts': counts,
            'raw_rates': {
                name: count / transitions if transitions else 0.0
                for name, count in counts.items()
            },
            f'{label}_to_teleport_ratio': (
                counts[label] / counts['teleport']
                if counts['teleport']
                else None
            ),
            'loader': loader_window_exposure(
                masks,
                offsets,
                lengths,
                frameskip=frameskip,
                num_steps=num_steps,
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument(
        '--environment', choices=('tunnel',), required=True
    )
    parser.add_argument('--frameskip', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=4)
    parser.add_argument('--teleport-threshold', type=float, default=30.0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit_dataset(
        args.dataset,
        environment=args.environment,
        frameskip=args.frameskip,
        num_steps=args.num_steps,
        teleport_threshold=args.teleport_threshold,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
