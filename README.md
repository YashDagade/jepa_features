# JEPA Features

`jepa_features` is the research codebase for studying **when predictive
world models preserve a feature and when their learned representation
abstracts it away**.

The first two controlled benchmarks are:

- **Tunnel Rooms Color:** ordinary motion is dominated by approaching a
  portal, while a comparatively rare crossing reveals the color-conditioned
  teleport rule.
- **Chaser Cue:** ordinary motion is dominated by chasing a moving runner,
  while contact reveals the runner-color-conditioned teleport rule.

The baseline matrix uses each model's native global representation. LeWM,
PLDM, and LpWM select the backbone's actual CLS token; Sensorimotor World
Model uses the same image-and-action contract. Each family retains only its
published training objective.

## Layout

- `stable-worldmodel/`: models, environments, collection, probes, decoders,
  and rollout diagnostics.
- `sbatch_scripts/`: short collection, training, and evaluation entrypoints.
- `paper/NeurIPS Workshop/`: anonymous workshop manuscript and frozen figure
  inputs.
- `.agents/`: cluster, code-quality, and scientific-scope notes.
- `datasets/` and `runs/`: ignored local mount points; large artifacts are
  never committed.

## Environment

On the Duke cluster, use:

```bash
source /scratch/yd3062/conda_envs/stable_worldmodel_latest/bin/activate
cd stable-worldmodel
pip install -e .
```

The launchers accept explicit external dataset paths, so existing HDF5 files
can be reused without copying them into this repository.

## Baseline question

For every model and dataset, report the dominant smooth-dynamics exposure,
rare teleport exposure, decoder cue fidelity, stage-specific linear probes,
and predicted transition behavior. A model is not called cue-blind from
planning success alone: the conclusion must localize loss to the encoder,
projector, predictor, or downstream energy.
