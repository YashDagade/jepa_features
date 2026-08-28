"""Summarize Tunnel Rooms HDF5 episode lengths and audit metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def _stats(lengths: np.ndarray) -> dict[str, Any]:
    is_integer = np.issubdtype(lengths.dtype, np.integer)
    min_value = lengths.min() if lengths.size else 0
    max_value = lengths.max() if lengths.size else 0
    if lengths.size:
        values, counts = np.unique(lengths, return_counts=True)
        mode_raw = values[np.argmax(counts)]
        mode_value = int(mode_raw) if is_integer else float(mode_raw)
    else:
        mode_value = 0
    return {
        'episodes': int(lengths.size),
        'frames': int(lengths.sum()) if is_integer else float(lengths.sum()),
        'mean': float(lengths.mean()) if lengths.size else 0.0,
        'std': float(lengths.std()) if lengths.size else 0.0,
        'min': int(min_value) if is_integer else float(min_value),
        'mode': mode_value,
        'p10': float(np.percentile(lengths, 10)) if lengths.size else 0.0,
        'p25': float(np.percentile(lengths, 25)) if lengths.size else 0.0,
        'p50': float(np.percentile(lengths, 50)) if lengths.size else 0.0,
        'p75': float(np.percentile(lengths, 75)) if lengths.size else 0.0,
        'p90': float(np.percentile(lengths, 90)) if lengths.size else 0.0,
        'iqr': float(np.percentile(lengths, 75) - np.percentile(lengths, 25))
        if lengths.size
        else 0.0,
        'max': int(max_value) if is_integer else float(max_value),
    }


def _load_lengths(dataset_cache_dir: Path, name: str) -> np.ndarray:
    path = dataset_cache_dir / 'datasets' / f'{name}.h5'
    with h5py.File(path, 'r') as file:
        return file['ep_len'][:]


def _dataset_terminal_summary(
    dataset_cache_dir: Path, name: str
) -> dict[str, Any]:
    path = dataset_cache_dir / 'datasets' / f'{name}.h5'
    with h5py.File(path, 'r') as file:
        ep_len = file['ep_len'][:].astype(np.int64)
        ep_offset = file['ep_offset'][:].astype(np.int64)
        end = ep_offset + ep_len - 1
        terminated = file['terminated'][end].astype(bool)
        truncated = file['truncated'][end].astype(bool)
    return {
        'terminated_episodes': int(terminated.sum()),
        'truncated_episodes': int(truncated.sum()),
        'success_rate': float(terminated.mean() * 100.0)
        if terminated.size
        else 0.0,
    }


def _scenario_digest(row: dict[str, Any]) -> str:
    payload = {
        key: row.get(key)
        for key in (
            'pairing',
            'endpoint_positions',
            'agent_position',
            'target_position',
            'target_leaf',
        )
    }
    encoded = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha1(encoded).hexdigest()


def _audit_summary(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    result: dict[str, Any] = {
        'num_paired_scenarios': manifest.get('num_paired_scenarios'),
        'dataset_names': manifest.get('dataset_names'),
        'scenario_constraints': manifest.get('scenario_constraints', {}),
        'dataset_summaries': manifest.get('dataset_summaries', {}),
    }
    for condition in ('hidden', 'paired_color'):
        rows = [
            row
            for row in manifest.get('episodes', [])
            if row.get('condition') == condition
        ]
        if not rows:
            continue
        steps = np.array([row.get('steps', 0) for row in rows], dtype=np.int32)
        teleports = np.array(
            [row.get('teleport_count', 0) for row in rows], dtype=np.int32
        )
        route_lengths = np.array(
            [row.get('route_length', 0.0) for row in rows], dtype=np.float32
        )
        digests = {_scenario_digest(row) for row in rows}
        ranks = sorted(
            {
                int(row['target_rank'])
                for row in rows
                if row.get('target_rank') is not None
            }
        )
        result[condition] = {
            'success_rate': float(
                np.mean([bool(row.get('success')) for row in rows]) * 100.0
            ),
            'target_rank_counts': {
                str(rank): int(
                    sum(
                        int(row.get('target_rank', -1)) == rank for row in rows
                    )
                )
                for rank in ranks
            },
            'steps': _stats(steps),
            'route_length': _stats(route_lengths),
            'teleport_count': _stats(teleports),
            'unique_scenarios': int(len(digests)),
            'duplicate_scenarios': int(len(rows) - len(digests)),
            'speed': sorted({float(row.get('speed', 0.0)) for row in rows}),
        }
    return result


def _plot(output_dir: Path, lengths_by_label: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, lengths in lengths_by_label.items():
        ax.hist(
            lengths,
            bins=40,
            density=True,
            histtype='step',
            linewidth=2,
            label=f'{label} (mean {lengths.mean():.2f})',
        )
    ax.set_title('Tunnel Rooms trajectory-length distribution')
    ax.set_xlabel('Stored frames per episode')
    ax.set_ylabel('Episode density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'trajectory_length_distribution.png', dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--hidden-name', default='tunnel_room_fast')
    parser.add_argument('--color-name', default='tunnel_room_color_fast')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lengths = {
        'Tunnel Rooms hidden fast': _load_lengths(
            args.dataset_cache_dir, args.hidden_name
        ),
        'Tunnel Rooms paired color fast': _load_lengths(
            args.dataset_cache_dir, args.color_name
        ),
    }
    summary = {
        'datasets': {
            label: {
                **_stats(values),
                **_dataset_terminal_summary(
                    args.dataset_cache_dir,
                    args.hidden_name if 'hidden' in label else args.color_name,
                ),
            }
            for label, values in lengths.items()
        },
        'audit': _audit_summary(args.manifest),
    }
    (args.output_dir / 'dataset_summary.json').write_text(
        json.dumps(summary, indent=2)
    )
    _plot(args.output_dir, lengths)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
