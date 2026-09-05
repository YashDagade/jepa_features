"""Map squared projected-state distance to fixed goal observations.

These are static encoder/projector comparisons, not predictive rollout costs.
Only valid agent centers are encoded; floor interpolation is visualization only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import stable_worldmodel as swm
from geometry import (
    build_grid,
    common_references,
    interpolate_to_canvas,
    make_env,
    preprocess_images,
)


def encode(model, images, device):
    """Encode observations in batches, retaining the projected planning state."""
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), 64):
            pixels = preprocess_images(images[start : start + 64]).to(device)
            features = model.encode({'pixels': pixels})['emb'][:, 0]
            if features.ndim != 2 or not torch.isfinite(features).all():
                raise ValueError(
                    'Expected finite frame-level projected states.'
                )
            batches.append(features.double().cpu().numpy())
    return np.concatenate(batches)


def distances(features, reference_count):
    """Return squared Euclidean distance and cosine similarity to each goal."""
    reference = features[:reference_count]
    queries = features[reference_count:]
    squared = ((queries[None] - reference[:, None]) ** 2).sum(-1)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-12)
    cosine = np.clip(
        normalized[:reference_count] @ normalized[reference_count:].T, -1, 1
    )
    return squared, cosine


def save_overlays(env, references, fields, output):
    """Share one scale across goals and preserve the environment's cue pixels."""
    from matplotlib import colormaps

    low, high = np.nanmin(fields), np.nanmax(fields)
    background = np.asarray(
        env.variation_space['appearance']['background'].value, dtype=np.uint8
    )
    for i, reference in enumerate(references):
        field = interpolate_to_canvas(
            env, fields[i], set(range(env.num_leaves + 1))
        )
        scaled = np.nan_to_num((field - low) / max(high - low, 1e-12))
        heat = colormaps['viridis_r'](scaled)[..., :3] * 255
        frame = env._render_frame(reference.position)
        floor = np.isfinite(field) & np.all(frame == background, axis=-1)
        frame[floor] = 0.15 * frame[floor] + 0.85 * heat[floor]
        Image.fromarray(frame).save(output / f'goal_{i + 1}_energy.png')
    return [float(low), float(high)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--scenario', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/energy'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--grid-size', type=int, default=20)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error('--grid-size must be at least 2')
    scenario = json.loads(args.scenario.read_text())
    env = make_env(scenario)
    try:
        references = common_references(scenario)
        grid = build_grid(env, args.grid_size)
        images = [env._render_frame(ref.position) for ref in references]
        images += [env._render_frame(pos) for _, _, pos in grid]
        if args.validate_only:
            print(
                json.dumps(
                    {
                        'valid_centers': len(grid),
                        'grid_size': args.grid_size,
                        'reference_positions': [
                            ref.position.tolist() for ref in references
                        ],
                    }
                )
            )
            return
        if args.checkpoint is None:
            parser.error(
                '--checkpoint is required unless --validate-only is set'
            )
        args.output.mkdir(parents=True, exist_ok=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = (
            swm.wm.load_pretrained(str(args.checkpoint)).to(args.device).eval()
        )
        model.requires_grad_(False)
        features = encode(model, images, args.device)
        squared, cosine = distances(features, len(references))
        shape = (len(references), args.grid_size, args.grid_size)
        squared_grid = np.full(shape, np.nan)
        cosine_grid = np.full(shape, np.nan)
        for i, (y, x, _) in enumerate(grid):
            squared_grid[:, y, x] = squared[:, i]
            cosine_grid[:, y, x] = cosine[:, i]
        np.savez_compressed(
            args.output / 'energy.npz',
            squared_projected_distance=squared_grid,
            cosine=cosine_grid,
            features=features,
            grid_indices=np.asarray([(y, x) for y, x, _ in grid]),
            grid_positions=np.stack([p for _, _, p in grid]),
            reference_positions=np.stack([ref.position for ref in references]),
        )
        scale = save_overlays(env, references, squared_grid, args.output)

        def digest(path):
            result = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    result.update(chunk)
            return result.hexdigest()

        metadata = {
            'checkpoint_sha256': digest(args.checkpoint),
            'model_config_sha256': digest(
                args.checkpoint.parent / 'config.json'
            ),
            'scenario_sha256': digest(args.scenario),
            'energy_sha256': digest(args.output / 'energy.npz'),
            'grid_size': args.grid_size,
            'valid_centers': len(grid),
            'encoding_dtype': 'float32',
            'distance_dtype': 'float64',
            'method': 'static squared projected-state distance; no predictor',
        }
        (args.output / 'provenance.json').write_text(
            json.dumps(metadata, indent=2) + '\n'
        )

        (args.output / 'scenario.json').write_text(
            json.dumps(scenario, indent=2) + '\n'
        )
        (args.output / 'scale.json').write_text(
            json.dumps({'squared_distance': scale, 'low_cost_color': 'yellow'})
            + '\n'
        )
    finally:
        env.close()


if __name__ == '__main__':
    main()
