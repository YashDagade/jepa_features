# Experiment launchers

Keep durable launchers short and explicit. Generated Slurm files, logs, and
resolved run metadata belong under ignored `runs/`; they are evidence, not
source code.

Every durable launcher must:

1. use `set -euo pipefail`;
2. record the exact Git commit, dataset path and SHA-256, seed, update budget,
   batch size, W&B project, group, run name, and run ID;
3. refuse a dirty source tree for real submissions;
4. verify the dataset hash before Python starts;
5. syntax-check every generated shell file with `bash -n`;
6. keep model and dataset sweeps in compact arrays; and
7. print all submitted job IDs for dependency and recovery audits.

The native baseline has one representation contract: the backbone's actual
CLS token. Do not add representation switches or aggregation overrides to a
launcher. LeWM, PLDM, and LpWM use no auxiliary inverse-dynamics option;
SMWM retains the inverse objective that defines that published method.

After submission, inspect the first log from every model family. A healthy job
has imported the committed checkout, verified the expected dataset hash,
allocated a GPU, produced finite losses, completed backward/optimizer steps,
and initialized the intended W&B run. Scheduler `Priority` is a queue state,
not a model failure.
