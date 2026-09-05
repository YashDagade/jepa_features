# LeWM cue binding in Tunnel Rooms

Anonymous code supplement for **When LeWM Misses the Color: Cue Binding in
Tunnel Rooms**. The code supports the Color, Shape, and green/blue Palette
conditions, LeWM training and readout controls, coordinate and image-decoder
probes, static goal energies, and rank-based planning.

## Installation

Python 3.12 is the tested runtime. From this directory:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install -e './stable-worldmodel[train,format]'
```

The requirements record tested direct dependency versions. They are not a
complete transitive lockfile. LeWM training uses one CUDA GPU. Collection,
coordinate probes, and energy diagnostics also support CPU execution. No
external experiment-tracking account is required; W&B is disabled by default.

## Data and training

Follow [datasets/TUNNEL_ROOMS.md](datasets/TUNNEL_ROOMS.md) to collect trajectories,
derive all three renderings, and construct episode-disjoint probe sidecars.
Datasets and trained weights are not bundled.

Run the following from `stable-worldmodel`, with `STABLEWM_HOME` pointing to the
output root used for collection:

```bash
python scripts/train/lewm_idm.py \
  feature_mode=global_spatial_8x8 data=tunnel_room_two_color \
  subdir=color-spatial-idm
```

Use `data=tunnel_room_two_shape` or `data=tunnel_room_two_green_blue` for the other
conditions. The primary configuration uses IDM weight 3.125. Controls use
`feature_mode=global_cls` or `loss.idm.weight=0`. To train native LeWM without an
allocated IDM head, use `scripts/train/lewm.py` with the same data and readout
options. An IDM head with weight zero and no allocated IDM head are separate
configurations. The implementation preserves shared-module initialization.

Portable weights and resolved model configuration are saved together under
`$STABLEWM_HOME/checkpoints/<subdir>/`. Decoder montages are saved locally under its `decoder/`
subdirectory. The detached decoder does not update the world model. Coordinate
probe commands and endpoint definitions are in the dataset guide. To enable
probes during training, set `TUNNEL_ROOMS_PROBE_SIDECAR` to the corresponding
sidecar path and add `epoch_probe=tunnel_color` (the callback supports all three
cue variants through their recorded endpoint names).

## Planning and static goal energies

These commands also run from `stable-worldmodel`. Validate the bundled Palette
rank scenarios without loading a model:

```bash
python scripts/plan/evaluate.py \
  --manifest scripts/plan/paper_palette_ranks.json --validate-only
```

Evaluate a trained Palette checkpoint using the training corpus for action
normalization:

```bash
python scripts/plan/evaluate.py --solver cem \
  --checkpoint /path/to/weights_epoch_10.pt \
  --dataset "$STABLEWM_HOME/datasets/tunnel_room_palette.h5" \
  --manifest scripts/plan/paper_palette_ranks.json \
  --output outputs/palette-cem
```

Select `--solver gd` for gradient descent. Color uses
`scripts/plan/paper_color_ranks.json`. Shape uses `scripts/plan/shape_ranks.json`
and `--identity paired_shape`; its manifest derives Shape rendering from the
archived Color geometry. A manifest must match the condition's rendering. Results include every
case, per-rank success, normalization statistics, and rollout videos. Solvers
use the paper's action-block and receding-horizon settings.

```bash
python scripts/analysis/energy.py \
  --checkpoint /path/to/weights_epoch_10.pt \
  --scenario scripts/analysis/paper_palette_scenario.json \
  --output outputs/palette-energy
python scripts/analysis/geodesic.py \
  --scenario scripts/analysis/paper_palette_scenario.json \
  --samples outputs/palette-energy/energy.npz \
  --output outputs/geodesic
```

There are corresponding `paper_color_scenario.json` and
`paper_shape_scenario.json` files. Energy arrays retain all queried positions;
PNG overlays share a scale across goals. Geodesics use the true portal pairings
and movement rules. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for interpretation
and the boundary between these runnable workflows and the archived experiments.

## Checks and release

```bash
cd stable-worldmodel
python -m pytest tests -q
python -m ruff check stable_worldmodel scripts
```

The manuscript's active LaTeX is retained in the branch under `paper/`, and is
submitted separately. To create the code ZIP from a clean, committed checkout,
run from the repository root:

```bash
python tools/package_submission.py --output /path/to/pta-code.zip
```

The exporter includes source, configs, tests, documentation, and checksums. It
excludes paper files, Git history, working directories, datasets, checkpoints,
and tracking metadata, and checks source text for personal paths and contact
links. Use this archive as the code upload. The upstream license is preserved;
see [NOTICE.md](NOTICE.md).
