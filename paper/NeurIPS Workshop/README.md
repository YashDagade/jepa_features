# PTA short paper

The anonymous manuscript is **Learning Where, but Not Reliably Which: Cue
Binding in a JEPA World Model**. Its scope is a controlled Tunnel Rooms case
study, using coordinate probes (MAE and R²), detached pixel decoders, and
rank-stratified planning. R² belongs to the coordinate probes, not the pixel
decoders.

The primary comparison uses native CLS selection with **LeWM+IDM 3.125**
for both completed Color and Shape conditions. This is an augmented training
objective, not native LeWM. The no-IDM and spatial-readout ablations are in the
appendix. Existing no-IDM Color and green/blue CLS runs both fail location and
cue probes; the green/blue result is therefore not a palette-specific anomaly
relative to a matched no-IDM Color control. All results remain one-seed cases.

A six-condition confirmation study was submitted from immutable source
revision `c4a2df2` on September 4, 2026: three cues, each with native LeWM and LeWM+IDM. Its values must not
be inferred or inserted before completed artifact checks. The matched
**green/blue CLS+IDM** row is still unavailable in this working draft, and
claims of three-cue CLS+IDM replication must await it. The jobs are
16970286/16970287/16970288 for Color/Shape/green-blue +IDM and
16970289/16970290/16970291 for the corresponding native no-IDM controls.
Generated scripts and outputs are under ignored `runs/lewm-cue-ablation-v1/`. The existing historical
Color/Shape rows are retained as measured evidence. No paper submission has
been made.

The transition-exposure pilot uses a spatial readout and improves selected
decoder appearances without consistent coordinate-probe recovery; it also
changes compute. It is labeled separately from the main CLS results and is
not claimed to demonstrate a remedy.

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

`data/short_paper_probe_results.csv` contains all six completed rows, including
the September 4 Shape and green/blue CLS follow-ups. Their retained features
independently reproduce archived cue scores; Shape location scores are
reconstructed under the same fixed ridge protocol from an aligned sidecar.
`short_paper_cls_probe_targets.csv` and `short_paper_cls_probe_provenance.csv`
record target-level results, hashes, and reconstruction checks.

Probe fitting and testing use disjoint episodes from the world-model corpus;
these are readout tests, not evidence of generalization to unseen training
episodes. Planning and decoder diagnostics have separate populations and
protocol boundaries, described in the paper. All main experiments have one
training seed. The completed follow-ups replace the former unverified cells. Compilation
is self-contained, but the newly requested uniform-IDM palette comparison
remains pending as described above.

## Artifacts

The tracked vector/preview figures and compact CSV inputs make the manuscript
self-contained for compilation. Three figures are in the main paper; the MAE
scale illustration is in the appendix. Large datasets, checkpoints, videos,
Slurm logs, audit tools, and figure builders remain under ignored `runs/` or
external storage. Builders can depend on those artifacts and are not claimed
to reproduce experiments from a clean clone. Logical artifact locators and
hashes preserve provenance without embedding workstation paths in the paper.

`research_memo.md` preserves broader evidence history, while `archive/`
contains superseded drafts; neither defines the short paper's current scope.
