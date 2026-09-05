"""Export the committed code supplement without Git history or local artifacts.

Run from a clean submission checkout. Only the allowlisted source tree enters
the archive. This tool also rejects common identifying paths and contact links.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    "README.md",
    "REPRODUCIBILITY.md",
    "NOTICE.md",
    "requirements-tested.txt",
    ".gitignore",
    "tools",
    "stable-worldmodel",
    "datasets",
}
BLOCKED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "wandb",
    "checkpoints",
    "outputs",
    "runs",
    ".venv",
}
IDENTIFIERS = {
    "machine-specific path": re.compile(
        r"/(?:scratch|home|state|Users)/[A-Za-z0-9_.-]+/"
    ),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "repository or tracking link": re.compile(
        r"(?:https?://|git@)(?:github\.com|gitlab\.com|wandb\.ai)[/:]"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def source_files():
    """Read committed regular files and reject identity-bearing text."""
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("Commit the reviewed changes before exporting.")
    files = []
    for entry in git("ls-tree", "-rz", "--full-tree", "HEAD").split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        path = PurePosixPath(raw_path.decode())
        if path.parts[0] not in TOP_LEVEL:
            continue
        if BLOCKED_PARTS.intersection(path.parts):
            raise ValueError(f"Generated or private path in source: {path}")
        if metadata.split()[0] not in (b"100644", b"100755"):
            raise ValueError(f"Only regular files may be exported: {path}")
        data = git("show", f"HEAD:{path}")
        if len(data) > 2_000_000:
            raise ValueError(f"Unexpectedly large source file: {path}")
        try:
            contents = data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"Unexpected binary source file: {path}") from None
        for label, pattern in IDENTIFIERS.items():
            if pattern.search(contents):
                raise ValueError(f"{label} found in {path}")
        files.append((str(path), data))
    if not files:
        raise RuntimeError("No source files selected.")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    files = source_files()
    checksums = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in files
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in [*files, ("SHA256SUMS", checksums.encode())]:
            info = zipfile.ZipInfo("pta-code/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{len(files)} source files; SHA-256 {digest}")


if __name__ == "__main__":
    main()
