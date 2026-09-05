# PTA short paper

The anonymous manuscript is **When LeWM Misses the Color: Cue Binding in Tunnel Rooms**. It introduces the controlled routing dataset and documents
a localization–cue-binding gap through affine probes (MAE and R²), detached
pixel decoders, static goal-distance maps, and rank-based planning. R² belongs
to coordinate probes, not pixel reconstructions.

The primary recipe is **LeWM with spatial 8×8 aggregation and IDM 3.125**.
Completed Color and Shape checkpoints align across probes, planning, and
static landscapes. Their decoder figures use the same training runs, but
show snapshots from the first batch of the final epoch rather than outputs
of the exported final checkpoint.
Location R² is .934/.907, versus cue R² .224/.203. Rank-R2 planning success is
16%/24% for CEM and 52%/40% for GD (Color/Shape). The main study concerns
cue binding in this LeWM recipe and its readout and inverse-dynamics
controls; it does not establish a universal JEPA failure.

CLS±IDM, alternate-palette spatial runs, and readout mechanics are in the
appendix. The six matched CLS runs completed on September 4, 2026 from
immutable source `c4a2df2`; they are separate from the historical primary
spatial checkpoints. Native CLS without IDM fails both location and cue
readouts. Spatial Color without IDM retains a selective gap, so IDM is not
necessary for that observation, but its causal influence is not ruled out.
The existing green/blue spatial runs use IDM weights 0/.625; they are not
presented as a matched 3.125 replication.

A targeted spatial confirmation campaign was submitted on September 4:
**16981833–16981836**, covering Color/Shape/green-blue + IDM3.125 and Color
without IDM. All four passed source/data checks and started. Their outputs
are under ignored `runs/lewm-spatial-confirmation-v1/`; no unfinished result
is included in the manuscript. This is a launch snapshot, not a live status
ledger. Existing CLS job IDs are 16970286–16970291.

The main energy figure changes the goal between both destination rooms for
the exact primary Color and Shape checkpoints. A third column shows a
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
planning literature; Appendix H develops the connections to LeJEPA/LpJEPA, identifiability,
slow features, auxiliary objectives, and value-guided planning. Appendix B credits Temporal
Straightening for the inspiration behind learned patch aggregation and states
the architectural and objective differences. The main narrative introduces the dataset, connects the four diagnostics,
and closes with the inconclusive exposure hypothesis and concrete next tests.
Readout sizes, loss weights, seeds, and diagnostic mechanics are confined to
the appendices; the main text identifies the LeWM adaptations in plain language.
`main.tex` contains this narrative; `appendix_readout_controls.tex`,
`appendix_evaluation.tex`, and `appendix_related_work.tex` contain the controls,
protocols and interpretation, and literature discussion respectively.
The main text ends on page four;
references follow the conclusion and span pages four through six, with the
appendix continuing below them. The complete PDF is 16 pages.

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

- `data/spatial_primary_probes.csv`, `spatial_primary_planning.csv`, and
  `spatial_primary_provenance.csv` freeze the aligned main results and
  distinguish serialized checkpoint-file hashes from canonical state hashes.
- `spatial_primary_decoder_sources.csv` records all six final-epoch clips at
  frame 2; `figures/lewm_spatial_primary_decoders.pdf` displays every clip.
- `lewm_spatial_similarity_maps.npz`, companion provenance and summary record
  newly evaluated static maps, using strictly loaded historical checkpoints
  and a dataset-verified renderer. The grid and reference images align with
  the separate CLS diagnostic protocol.
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

The appendix now includes all twelve TR2 conditions from the historical
factorial: CLS/spatial 8×8 crossed with IDM weights 0, .005, .025, .125, .625,
and 3.125. One identical input is used throughout, with unchanged cue crops.
The figure uses retained periodic training media; filename label 90001 is
not treated as an optimizer-step count or final-checkpoint identifier.
Spatial aggregation improves marker visibility even without IDM. The text
explains how IDM trains states to retain action-relevant information without
claiming that it directly supervises color or that both additions are necessary.

Historical `final_epoch_*` media used in Figure 2 are captured at the first
batch of the final epoch. The separate matched CLS GIFs instead come from
validation after epoch 10 training and align with those final model weights.
