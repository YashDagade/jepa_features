# NeurIPS Workshop draft

This anonymous workshop draft uses the supplied ICML 2026 two-column style.
The complete paper, including its appendix, is kept in `main.tex` so it can be
read and edited in one place. The current draft prespecifies the native-CLS
baseline and uses explicit pending-result markers; values should be inserted
only from the new repository's canonical manifests.

Run `make build` to compile the paper and refresh the ignored local
`main.pdf`. The build requires Tectonic 0.16.9 or newer on `PATH`:

```bash
make build
```

The Makefile exports a fixed `SOURCE_DATE_EPOCH`, so identical clean inputs
produce an identical PDF. For TeX-only live rebuilding, use `make watch`;
Tectonic's live output remains under `build/main/main.pdf` until the next
`make build`.

The `data/` and `figures/` directories preserve the audited tables and final
assets from the earlier LeWM color-blindness study. The prior complete draft is
kept read-only in `archive/color_blind_jepa_draft.tex`; it is evidence and prose
provenance, while `main.tex` is the active feature-abstraction manuscript.
