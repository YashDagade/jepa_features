# PTA short paper

The anonymous manuscript is **Learning Where, but Not Reliably Which: Cue
Binding in a JEPA World Model**. Its scope is a controlled Tunnel Rooms case
study, using coordinate probes (MAE and R²), detached pixel decoders,
static goal-distance landscapes, and rank-stratified planning. R² belongs to the coordinate probes, not the pixel
decoders.

The primary comparison uses native CLS selection with **LeWM+IDM 3.125**
across matched Color, Shape, and green/blue conditions. This is an augmented
training objective, not native LeWM. All six conditions (three cues, with and
without IDM) completed successfully from immutable revision `c4a2df2` on
September 4, 2026. The matched no-IDM results and historical spatial-readout
controls are reported separately in the appendix. All results remain
one-seed cases.

The primary probe table and three-column decoder figure now use these new
matched checkpoints. All six final affine probe results reproduce exactly
from retained frozen features. Dataset, sidecar, configuration, checkpoint
file, result and feature hashes are preserved in the matched provenance CSV.
Historical planning, hue-classification and transition-exposure evidence is
explicitly distinguished because it uses different checkpoints.

Jobs 16970286/16970287/16970288 cover Color/Shape/green-blue +IDM and
16970289/16970290/16970291 their native no-IDM controls. Generated scripts and
outputs are under ignored `runs/lewm-cue-ablation-v1/`. No paper submission
has been made.

The main argument now follows cue-conditioned readouts, reconstructions,
rank-based planning, and a six-panel goal-energy figure that changes the goal
between both destination rooms for all three matched cue variants. These
energies compare encoded observations, not predicted action outcomes. The
conclusion summarizes the observed gap and identifies controlled follow-up tests.

The transition-diversity ablation is consolidated in the appendix. It reports
more saturated selected decoder examples without consistent coordinate-probe
recovery, together with the exposure counts and compute confound. The main
text retains only the rare-crossing hypothesis and an appendix pointer; the
abstract contains no transition-intervention claim.

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

Run from this directory:

```bash
make build
```

The build requires Tectonic 0.16.9 or newer on `PATH`. The Makefile fixes
`SOURCE_DATE_EPOCH`; identical clean inputs therefore produce identical PDF
bytes. The current PDF is `main.pdf` (generated and ignored by Git).
`make watch` supports TeX-only live rebuilding under `build/main/`.

## Evidence and completed follow-ups

`data/short_paper_matched_probe_results.csv` contains the six new completed
conditions and `short_paper_matched_probe_provenance.csv` records their
verification. `short_paper_cls_decoder_sources.csv` identifies the three
matched epoch-10 validation montages used in Figure 2: frame 2, all six clips,
with only empty montage padding removed. Historical tables and source files
are retained under their original names for the supporting ablations.

Probe fitting and testing use disjoint episodes from the world-model corpus;
these are readout tests, not evidence of generalization to unseen world-model
training episodes. Planning and decoder diagnostics have separate populations
and protocol boundaries, described in the paper. No new planning scores are
claimed for the six matched checkpoints.

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
