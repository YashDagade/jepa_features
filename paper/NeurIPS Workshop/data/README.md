# Paper data

This directory contains compact, tracked numeric sources for the manuscript.
Large datasets, checkpoints, decoder videos, and raw W&B exports remain outside
the paper tree.

Files added for the LeWM-focused revision:

- `lewm_alternate_palette_probe.csv` records the matched green/orange and
  green/blue global localization controls at auxiliary weights 0 and .625. The
  standard rows come from the historical LeWM color-identity factorial and the
  alternate rows from the audited blue/green control. The layouts, actions,
  targets, rewards, and episode structure match across 10,000 scenarios after
  excluding `tunnel_colors`, `tunnel_color_names`, and the palette-dependent
  `scenario_hash`.
- `lewm_palette_decoder_summary.csv` records the held-out green/orange decoder
  audit and the smaller logged-media summaries for the standard and blue/green
  palettes. The `analysis_protocol` column distinguishes the prespecified
  episode audit from the weaker media-support summaries.
- `metric_sensitivity.csv` contains the six-setting standard-palette decoder
  support-means used for the 4.0--7.5% descriptive range in the main text. The
  held-out E024 audit in `lewm_palette_decoder_summary.csv` is the stronger
  inferential result.
- `lewm_planning_cue_ablation.csv` records the completed historical LeWM
  visible-versus-hidden Color and Shape planning evaluations. Every row includes
  the exact W&B run identifier, project, seed, solver, rank, and episode count.
  These are one-seed, separately trained models with a historical
  position-aware global state and auxiliary inverse-dynamics weight 3.125.
- `lewm_planning_protocols.csv` records source commits, checkpoint epochs,
  manifest hashes, and the within-pair matching scope. Color uses exact shared
  manifests. Shape uses condition-specific manifests with matched case
  identifiers and geometry.

The verified alternate palette is green/blue. Red is the agent color in the
rendered data; no red/green portal experiment was recovered from the tracked
repository, historical worktrees, manifests, or relevant W&B projects.

`shape_followup_prespecification.csv` is a historical launch-time record. Its
status strings describe the state when the file was written and should not be
used as a current scheduler ledger.
