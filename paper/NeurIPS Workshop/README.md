# NeurIPS 2026 short workshop paper

This anonymous 4-page extended abstract studies one narrow finding in Tunnel
Rooms: LeWM's compact predictive state preserves portal location much better
than cue-conditioned portal identity. The paper uses only LeWM evidence and
triangulates the finding with affine probes, rank-stratified planning, and
detached decoders. It explicitly distinguishes cue underuse in the projected
state from early visual perception: native patch tokens retain visible hue.

`main.tex` uses the official unmodified NeurIPS 2026 style in
double-blind-workshop mode for the workshop *From Pretrained Representations to
Acting Agents*. The complete argument is contained in four content pages;
references and optional technical appendices follow.

## Build

Run from this directory:

```bash
make build
```

The build requires Tectonic 0.16.9 or newer on `PATH`. The Makefile fixes
`SOURCE_DATE_EPOCH`; identical clean inputs therefore produce identical PDF
bytes. `make watch` supports TeX-only live rebuilding under `build/main/`.

## Figure sources

The final vector and preview figures are tracked in `figures/`, and their
compact numerical sources are tracked in `data/`. Figure builders are local
development utilities under ignored `runs/` directories; they depend in part
on large external checkpoints and datasets and are not presented as clean-clone
reproduction commands. The tracked PDF/PNG assets are self-contained for paper
compilation.

## Evidence layout

- `data/` contains compact machine-readable values and experiment records for
  every active quantitative figure and table.
- `figures/` contains the active MAE-scale, three-cue decoder, and
  transition-exposure figures in PDF and PNG form. LaTeX uses the vector PDFs;
  PNGs support quick visual inspection.
- `data/short_paper_*.csv` contains the active compact table values and exact
  decoder provenance. Older tables remain as supporting audit records.
- `research_memo.md` records the broader evidence history; it is not the scope
  of the short paper.
- `archive/` preserves superseded drafts and is not included by `main.tex`.

The manuscript intentionally omits workstation paths, account URLs, and long
hash ledgers. Anonymous logical artifact locators and hashes remain in the
supplementary CSV records, while large checkpoints, datasets, videos, and raw
action tensors remain outside the tracked paper tree.
