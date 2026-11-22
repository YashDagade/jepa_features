# Tunnel Room: Probing What JEPA World Models Preserve

Camera-ready LaTeX source, vector figures, and frozen experimental results for:

> **Tunnel Room: Probing What JEPA World Models Preserve**  
> Yash Dagade\* (Pantheon) and Taj Gillin (Brown University)  
> *NeurIPS 2026 Workshop: From Pretrained Representations to Acting Agents (PTA)*  
> \*Work done while Member of Technical Staff at Pantheon.

## Structure

- `main.tex` — Four-page main paper, acknowledgments, references, and appendix includes (`neurips_2026.sty` in `[dblblindworkshop,final]` mode).
- `appendix_variants.tex`, `appendix_readout_controls.tex`, `appendix_evaluation.tex`, `appendix_followup_diagnostics.tex`, `appendix_related_work.tex` — Technical appendices.
- `references.bib` — Bibliography (23 cited entries).
- `figures/` — The 9 vector figure PDFs included in the manuscript.
- `data/` — Frozen CSV/NPZ/JSON experimental data backing tables and figures.

## Build

From this directory:

```bash
make build
```

This compiles `main.tex` with `pdflatex` + `bibtex` and writes `main.pdf`. To build with Tectonic (v0.16.9+) instead:

```bash
make build-tectonic
```
