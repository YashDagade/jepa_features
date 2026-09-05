#!/usr/bin/env python3
"""Rebuild main.pdf whenever local TeX sources change."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUFFIXES = {'.tex', '.bib', '.sty'}


def source_stamp() -> dict[Path, float]:
    return {
        path: path.stat().st_mtime
        for path in ROOT.iterdir()
        if path.suffix in SUFFIXES
    }


def main() -> None:
    previous: dict[Path, float] = {}
    print(f'watching {ROOT}', flush=True)
    while True:
        current = source_stamp()
        if current != previous:
            previous = current
            print('rebuilding main.pdf ...', flush=True)
            code = subprocess.call(['make', 'build'], cwd=ROOT)
            print('ok' if code == 0 else 'build failed', flush=True)
        time.sleep(0.75)


if __name__ == '__main__':
    main()
