# NeurIPS workshop draft

This anonymous draft develops one claim: task-relevant information can be
linearly accessible without inducing the action ordering required by latent
planning. `main.tex` follows the chain from ordered patch features to the native
global state, action-conditioned prediction, native cost, and numerical search.

The supplied venue style is still provisional: the paper uses the ICML 2026
two-column package while awaiting a confirmed workshop template. The current
build has 9 pages of main text and 16 pages total, including references and the
appendix. A venue-specific formatting pass and deliberate length decision are
still required before submission.

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
- `figures/` contains five main figures and two appendix figures in PDF and PNG
  form. LaTeX uses the vector PDFs; PNGs support quick visual inspection.
- `research_memo.md` records the evidence-backed argument, experiment
  boundaries, and next interventions.
- `archive/` preserves superseded drafts and is not included by `main.tex`.

The manuscript intentionally omits workstation paths, account URLs, and long
hash ledgers. Anonymous logical artifact locators and hashes remain in the
supplementary CSV records, while large checkpoints, datasets, videos, and raw
action tensors remain outside the tracked paper tree.
