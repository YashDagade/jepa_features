# Cluster workflow

Use the shared runtime for tests, collection, training, and W&B access:

```bash
PYTHON=/scratch/yd3062/conda_envs/stable_worldmodel_latest/bin/python
```

## Slurm

- Keep durable launchers short and submit one generated job per run.
- Resolve repository path and Git commit into `run_info.txt` before launch.
- Syntax-check generated shell with `bash -n`.
- Treat `Priority`, `QOSGrpGRES`, and `QOSMaxGRESPerUser` as scheduling states,
  not training failures.
- Confirm health from the first log: imports, dataset hash, CUDA allocation,
  finite loss, backward progress, checkpoint path, and W&B run ID.
- Research diagnostics fail loudly. Missing probes or media must not be
  silently converted to zero or success.

## Storage

Datasets and checkpoints stay on scratch and are ignored by Git. Never place
large HDF5 files, W&B caches, generated videos, checkpoints, or Slurm logs in a
commit. Final compact tables, plots, TeX, and reproducibility manifests may be
tracked.

## Git

Commit coherent, passing milestones. Do not mix unrelated experiments into a
commit. Before pushing, run focused tests, Ruff, `git diff --check`, a forbidden
configuration-name audit, and a tracked-file size audit.
