from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

SKIP_SUFFIXES = {
    ".7z",
    ".a",
    ".class",
    ".dll",
    ".dylib",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".lock",
    ".mp3",
    ".mp4",
    ".o",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".svg",
    ".tar",
    ".tgz",
    ".wasm",
    ".webp",
    ".zip",
}


def iter_searchable_files(paths: list[str], *, include_hidden: bool = False) -> Iterator[Path]:
    seen: set[Path] = set()

    for raw_path in paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.exists():
            continue

        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in seen and _is_searchable_file(candidate):
                seen.add(resolved)
                yield candidate
            continue

        for root, dirnames, filenames in os.walk(candidate):
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if _include_dir(dirname, include_hidden=include_hidden)
            ]

            for filename in filenames:
                if not include_hidden and filename.startswith("."):
                    continue
                path = Path(root) / filename
                resolved = path.resolve()
                if resolved in seen or not _is_searchable_file(path):
                    continue
                seen.add(resolved)
                yield path


def _include_dir(dirname: str, *, include_hidden: bool) -> bool:
    if dirname in SKIP_DIR_NAMES:
        return False
    if not include_hidden and dirname.startswith("."):
        return False
    return True


def _is_searchable_file(path: Path) -> bool:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    return not _is_probably_binary(path)


def _is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError:
        return True

    if b"\x00" in sample:
        return True

    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return True

    return False
