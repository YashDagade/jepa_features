# Stable WorldModel research package

This directory contains the executable package used by `jepa_features`.
It is based on [Stable WorldModel](https://github.com/galilai-group/stable-worldmodel)
and keeps its data, environment, action-conditioned prediction, and solver
interfaces. The research additions retained here are limited to Tunnel Rooms,
Chaser, LpWM, Sensorimotor World Model, global decoders, stagewise probes, and
simulated rollout diagnostics.

## Install

```bash
pip install -e '.[train,format]'
```

On the shared cluster, invoke scripts with:

```bash
/scratch/yd3062/conda_envs/stable_worldmodel_latest/bin/python \
  scripts/train/lewm.py --help
```

## Native model contract

LeWM, PLDM, and LpWM select `last_hidden_state[:, 0]`, the ViT's actual CLS
token. The predictor receives actions through the models' native action
encoder and AdaLN conditioning.

Large datasets, checkpoints, W&B media, and generated cluster scripts live
outside Git. Use the root launchers to record their absolute paths and hashes.

The upstream package is MIT licensed; see `LICENSE`.
