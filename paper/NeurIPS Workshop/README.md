# NeurIPS Workshop draft

This anonymous workshop draft uses the supplied ICML 2026 two-column style.
The complete paper, including its appendix, is kept in `main.tex` so it can be
read and edited in one place. In this provisional style the current build has
19 pages of main text and 26 pages total; it is a full research draft, not yet
an 8--10-page or venue-template-compliant submission. Applying the actual
workshop template and making a deliberate main-text cut remain required before
submission. The current draft reports the completed native
50-epoch matrix, spatial Tunnel controls, dimension-matched checkpoint
patch readouts, a validated reconstructed-initialization control paired with
six globally seeded checkpoints at epochs 1 and 10, and exact Chaser/Tunnel cue
interventions. Numerical values are taken from the
repository's canonical manifests and checkpoint probe artifacts. Remaining
evidence gaps are marked with specific `\yash{TODO: ...}` comments.

Run `make build` to compile the paper and refresh the ignored local
`main.pdf`. The build requires Tectonic 0.16.9 or newer on `PATH`:

```bash
make build
```

The checkpoint-trajectory and decoder figures are reproducible from the
completed artifacts with the ignored analysis builder:

```bash
/scratch/yd3062/conda_envs/stable_worldmodel_latest/bin/python \
  runs/color-onset-v1/report/build_report.py
```

The builder writes its full visual report under `runs/color-onset-v1/report/`
and refreshes the paper-ready PDF/PNG assets in `figures/`. W&B media filenames
contain logger steps rather than epochs; the builder maps the chronologically
ordered validation media to completed epochs.

The Makefile exports a fixed `SOURCE_DATE_EPOCH`, so identical clean inputs
produce an identical PDF. For TeX-only live rebuilding, use `make watch`;
Tectonic's live output remains under `build/main/main.pdf` until the next
`make build`.

The `data/` and `figures/` directories preserve the audited tables and final
assets from the earlier LeWM color-blindness study. The prior complete draft is
kept read-only in `archive/color_blind_jepa_draft.tex`; it is evidence and prose
provenance, while `main.tex` is the active feature-abstraction manuscript.
`research_memo.md` records the evidence-backed thesis change, rejected
hypotheses, corrected DINO interpretation, and ranked next experiments.

`data/chaser_contact_null.csv` records the exact-sample contact-quadrant-only
destination baseline and episode-bootstrap sensitivity used in the revised
Chaser analysis. It is a cue-blind diagnostic null, not a model checkpoint.
`data/objective_exposure.csv` records the loader-contract audit that supports
the transition-exposure axes and counts.
`data/capacity_matched_tunnel_readouts.csv` records the 192-dimensional
CLS/mean/PCA/random-projection comparison and paired binding deltas from the
same four epoch-50 Tunnel encoders.
`data/capacity_matched_tunnel_trajectory.csv` records the corresponding
epoch-1/10/50 fixed-width trajectories.
`data/seeded_epoch0_1_10_capacity.csv` records the validated reconstructed
initialization and epoch-1/10 LeWM/SMWM seed-3072--3074 fixed-width control, and
`data/seeded_smwm_native_cls_epoch0_10.csv` records the post-hoc descriptive
native-CLS SMWM trajectories for those three seeds through the right-censored
epoch-10 horizon. It separates agent localization, nearest-tunnel geometry, and
color--place binding and must not be read as a sigmoid or population estimate.
`data/tunnel_exact_binding_swap.csv` records the exact Tunnel rendering-swap
stage responses and simultaneous decisions. The three Chaser checkpoint CSV files
record the nine-checkpoint causal-response family, destination-stage
diagnostics, and scheduled-onset decisions.
