# PTA short paper

The anonymous manuscript is **When LeWM Misses the Color: Cue Binding in Tunnel Room**. It introduces the controlled routing dataset and documents
a localization–cue-binding gap through affine probes (MAE and R²), detached
pixel decoders, static goal-distance maps, and rank-based planning. R² belongs
to coordinate probes, not pixel reconstructions.

The primary recipe is **LeWM with spatial 8×8 aggregation and IDM 3.125**.
Completed Color, Shape, and Palette checkpoints align across probes, planning, and
static landscapes. Their decoder figures retain all six clips per run. Color/Shape snapshots
come from the first batch of the final epoch; Palette uses validation after
the final epoch. The populations and capture times differ.
Location R² is .934/.907/.930, versus cue R² .224/.203/.258
(Color/Shape/Palette). Rank-R2 planning success is
16%/24%/28% for CEM and 52%/40%/32% for GD (Color/Shape/Palette). The main study concerns
cue binding in this LeWM recipe and its readout and inverse-dynamics
controls; it does not establish a universal JEPA failure.

CLS±IDM, alternate-palette spatial runs, and readout mechanics are in the
appendix. The six matched CLS runs completed on September 4, 2026 from
immutable source `c4a2df2`; they are separate from the historical primary
spatial checkpoints. Native CLS without IDM fails both location and cue
readouts. Spatial Color without IDM retains a selective gap, so IDM is not
necessary for that observation, but its causal influence is not ruled out.
The appendix now includes the historical Color 2×2 planning ablation, with
all 16 rank/solver results in `data/lewm_idm_readout_planning.csv`. For the
spatial readout, IDM raises success over 50 cases from 8% to 48% (CEM) and
8% to 58% (GD). Within each readout, the IDM pair shares initialization,
training budget, and evaluation cases/settings. Across readouts, the same
seed does not preserve encoder initialization because aggregator construction
consumes RNG first. Spatial aggregation improves location MAE from 7.42 to
3.83 pixels with IDM fixed; it does not restore cue binding. These one-seed
results motivate the configuration without claiming universal necessity.

The existing green/blue spatial runs use IDM weights 0/.625; they are not
presented as a matched 3.125 replication.

A targeted spatial confirmation campaign was submitted on September 4:
**16981833–16981836**, covering Color/Shape/green-blue + IDM3.125 and Color
without IDM. All four passed source/data checks and started. Their outputs
are under ignored `runs/lewm-spatial-confirmation-v1/`; no unfinished result
is included in the manuscript. This is a launch snapshot, not a live status
ledger. Existing CLS job IDs are 16970286–16970291.

The Palette extension uses the completed epoch-10 checkpoint from training
job **16981835** and final static-map job **16985126**. The independent probe
reconstruction matches all endpoint metrics within 1.42e-14. Final CEM/GD
planning job **16985224** uses the historical solvers and Color case geometry,
with the green/blue palette. Both solvers completed successfully: CEM R1/R2 is 80%/28%; GD is 84%/32%.
Final metric hashes and verified outcome counts are in
`spatial_primary_planning.csv`. Palette uses source `c4a2df2`,
so the three primary variants share the nominal recipe but are not a fully
controlled palette-only intervention. Saved rollout videos confirm that
historical Color evaluation uses filled markers matching training; the old
outline caveat belonged to a different checkpoint.

The main energy figure changes the goal between both destination rooms for
the exact primary Color, Shape, and Palette checkpoints. A fourth column shows a
geodesic navigation reference through the true portal pairings, including
triggering, landing offsets, and rearm state. It uses a one-pixel finite
movement graph and the same 210 plotted positions; half-pixel refinement
changes displayed distances by less than .900 px. This reference uses exact
point goals and pixel path length, not the simulator success region or
calibrated latent-energy units. Static energies compare
encoded observations, not predicted teleportation outcomes. The transition
exposure pilot remains in the appendix: selected reconstructions are more
saturated, but coordinate probes do not consistently improve and training
compute increases. The conclusion identifies concrete causal and replication
tests. No paper submission has been made.

## Submission handoff

The September 5 formatting audit is complete. See [SUBMISSION.md](SUBMISSION.md)
for the upload files, independently verified source archive, OpenReview fields,
and deadline. The final PDF contains only embedded Type 1/TrueType fonts;
tables use the official 10-point size. The updated source archive rebuilds a
byte-identical PDF. No OpenReview submission has been sent.

## Venue and format

Verified September 4, 2026 against the [PTA call for papers](https://ptaworkshop.github.io/call-for-papers.html):

- Short submissions allow up to four NeurIPS-format content pages.
- Review is double blind; the NeurIPS Paper Checklist is not required.
- The submission deadline is September 5, 2026, AoE.
- The CFP adopts the [official NeurIPS 2026 guidelines](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip), which exclude references and optional technical appendices from the content limit. Appendices belong in the same PDF and the main argument must stand alone.

`main.tex` uses the official, unmodified 2026 style in `dblblindworkshop` mode.
The work fits the workshop topics on diagnosing representation failures in
planning and evaluating actionable representations. No submission has been
made by the revision workflow.

## Build

The current draft cites **22 distinct references**, all attached to claims
in the text. The introduction motivates cue binding through the JEPA and latent
planning literature; Appendix J develops the connections to LeJEPA/LpJEPA, identifiability,
slow features, auxiliary objectives, and value-guided planning. Appendix B credits Temporal
Straightening for the inspiration behind learned patch aggregation and states
the architectural and objective differences. The main narrative introduces the dataset, connects the representation, prediction, and planning diagnostics,
and closes with the inconclusive exposure hypothesis and concrete next tests.
Readout sizes, loss weights, seeds, and diagnostic mechanics are confined to
the appendices; the main text identifies the LeWM adaptations in plain language.
`main.tex` contains this narrative; `appendix_readout_controls.tex`,
`appendix_evaluation.tex`, and `appendix_related_work.tex` contain the controls,
protocols and interpretation, and literature discussion respectively.
The main text ends on page four;
references follow the conclusion and span pages five through six, with the
appendix continuing below them. The complete PDF is 20 pages.

Layout uses natural vertical spacing, keeps the planning paragraph together,
and places tables near their discussion. Landscape figures are interleaved
with their methods; the historical atlas combines nine LeWM cases in one figure. References
use abbreviated author names and compact entry spacing. The official style,
body font and margins are preserved.

Run from this directory:

```bash
make build
```

The build requires Tectonic 0.16.9 or newer on `PATH`. The Makefile fixes
`SOURCE_DATE_EPOCH`; identical clean inputs therefore produce identical PDF
bytes. The current PDF is `main.pdf` (generated and ignored by Git).
`make watch` supports TeX-only live rebuilding under `build/main/`.

## Evidence and completed follow-ups

- `data/spatial_primary_probe_targets.csv` gives all individual endpoint MAE
  and R² values for current and predicted states. Table 1 displays each current
  endpoint separately, distinguishing central and destination rooms; the prose
  explicitly labels four-endpoint means.
- `data/spatial_primary_probes.csv`, `spatial_primary_planning.csv`, and
  `spatial_primary_provenance.csv` freeze the aligned main results and
  distinguish serialized checkpoint-file hashes from canonical state hashes.
- `spatial_primary_decoder_sources.csv` and `spatial_palette_decoder_sources.json`
  record all six clips per variant at frame 2;
  `figures/lewm_spatial_three_cue_decoders.pdf` displays every clip.
- `lewm_spatial_similarity_maps.npz`, companion provenance and summary record
  newly evaluated static maps, using strictly loaded historical checkpoints
  and a dataset-verified renderer. The grid and reference images align with
  the separate CLS diagnostic protocol. `lewm_spatial_palette_maps.npz` adds
  the verified Palette maps on the identical grid and reference positions;
  its dedicated static provenance records the later source revision.
- `short_paper_matched_probe_results.csv` and companion provenance retain the
  six completed CLS conditions. Every affine score reproduces exactly from
  retained frozen features. The three-cue CLS decoder figure remains in the
  appendix, including green/blue.

Probe fitting and testing use disjoint episodes from the world-model corpus;
these are readout tests, not generalization to episodes unseen during
world-model training. Planning and decoder diagnostics use separate
populations, even when the checkpoint agrees. The archived CEM horizon sweep
used older checkpoints and is not claimed as robustness evidence for the
primary spatial recipe. All reported training conditions use one seed.

## Artifacts

The tracked vector/preview figures and compact CSV inputs make the manuscript
self-contained for compilation. Task, decoder, and representation-landscape diagnostics are in the main
paper; transition-exposure and MAE illustrations are in the appendix. Large datasets, checkpoints, videos,
Slurm logs, audit tools, and figure builders remain under ignored `runs/` or
external storage. Builders can depend on those artifacts and are not claimed
to reproduce experiments from a clean clone. Logical artifact locators and
hashes preserve provenance without embedding workstation paths in the paper.

`research_memo.md` preserves broader evidence history, while `archive/`
contains superseded drafts; neither defines the short paper's current scope.

## Readout and IDM decoder sweep

The appendix displays six Tunnel Room conditions from the historical factorial:
CLS/spatial 8×8 crossed with IDM weights 0, .125, and 3.125. One identical
input is used throughout, with unchanged cue crops. The full twelve-condition
source arrays and provenance remain available.
The figure uses retained periodic training media; filename label 90001 is
not treated as an optimizer-step count or final-checkpoint identifier.
Spatial aggregation improves marker visibility even without IDM. The text
explains how IDM trains states to retain action-relevant information without
claiming that it directly supervises color or that both additions are necessary.

Historical `final_epoch_*` media used in Figure 2 are captured at the first
batch of the final epoch. The separate matched CLS GIFs instead come from
validation after epoch 10 training and align with those final model weights.

## September 5 feedback experiments

Both follow-ups are complete on main; the anonymous code branch and earlier
code ZIP remain a separate snapshot. Updated PDF and LaTeX upload files are
in `runs/pta_feedback/compact_planning_table/submission/`. The main paper remains four pages.

A validation-selected two-hidden-layer MLP uses the same frozen checkpoints
and endpoint targets as the main probes. Across Color/Shape/Palette, current
cue R² is .273/.261/.259, compared with location R² .885/.857/.874. The MLP
does not close the gap relative to a ridge baseline fit on the same 512
episodes. Three probe initializations quantify head variability, not variation
across world-model training runs. Saved-head replay, fit-only normalization,
validation selection, and all historical endpoint metrics passed independent
checks. Full endpoints, protocol, provenance, and the audit are in
`data/lewm_nonlinear_probe_*`.

The controlled swap experiment uses 16 evaluation geometries and all three
cue renderings. Swapping the central-room endpoints changes visible cues and
actual routing together. Encoders respond to this change; they are not cue
invariant. Under identical actions, observed latent separation increases after
teleportation, while open-loop forecasts understate it. Median predicted /
observed squared separation over the first three post-crossing samples is
.252/.415/.177. A separate training geometry illustrates the intervention and
is excluded from these aggregates. General forecast error remains an
alternative explanation. Compact per-case arrays, protocol, input/output
hashes and validation are in `data/lewm_portal_swap_*`; full experiment
artifacts and runners remain under ignored `runs/pta_feedback/`.


The main paper now explicitly distinguishes randomized spatial goals from
fixed temporal goal offsets. All primary evaluations start in the hub and
end at a sampled position in a randomly chosen leaf, with the goal image
rendered directly. The 25-primitive-step planning horizon does not determine
goal selection. `data/lewm_planning_goal_sampling_audit.json` verifies all
50 cases per variant and their shared geometry, starts, goals, and ranks.
Figure 3 combines the unchanged goal landscapes with portal-swap divergence
curves for all three variants. The main text explains the intervention,
pre-crossing initialization, static sensitivity, forecast attenuation, and
remaining alternative explanation; the full controls remain in Appendix I.

## Planning by required cue and first entrance

Table 2 uses one compact row per variant and two grouped headers:
rank-based performance (R1/R2), followed by cue-based performance (Cue 1/Cue 2).
Cue names are defined in the caption. Every value reports CEM/GD task success. The appendix
retains the complete cue-by-rank breakdown and case counts. All 300 episode records were joined
to their exact hashed manifests and checked against the original rank totals.
Every cue/solver has lower observed success on R2. Pooled cue success spans
44.1–75.0%; these groups are not rank-balanced (12/4 versus 13/21 R1/R2
cases), so pooled differences cannot establish a color or shape preference.

The appendix reports actual first hub-to-leaf teleports. On R2, CEM enters
the nearer, incorrect tunnel first in 22/25, 19/25, and 18/25 Color/Shape/Palette
cases; GD does so in 13/25, 14/25, and 18/25. No-entry cases remain in the
denominator. This supports a proximity bias, strongest for CEM, rather than
a universal nearest-tunnel policy. These are new analyses of completed runs,
not additional model evaluations.
