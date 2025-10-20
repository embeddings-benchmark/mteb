#!/usr/bin/env python3
"""Recursively rename .py files from CamelCase/PascalCase to snake_case,
preserving and correctly handling tokens like p2p, i2i, it2i, it2t, i2t, t2i, etc.

Examples:
    BLINKIT2IRetrieval.py -> blink_it2i_retrieval.py
    BLINKIT2TRetrieval.py -> blink_it2t_retrieval.py
    TaskP2PEncoder.py     -> task_p2p_encoder.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FIXED_TOKENS: list[str] = [
    "it2t",
    "it2i",
    "i2it",
    "t2it",
    "it2it",
    "p2p",
    "s2s",
    "s2p",
    "p2s",
    "t2t",
    "t2c",
    "i2i",
    "i2c",
    "i2t",
    "t2i",
]


def pre_normalize_tokens(name: str) -> str:
    """Insert underscores around known fixed tokens *before* any CamelCase splitting.

    Works even if the token spans uppercase boundaries, e.g.:
        BLINKIT2IRetrieval -> BLINK_it2i_Retrieval
    """
    sorted_tokens = sorted(FIXED_TOKENS, key=len, reverse=True)
    # Build a combined regex pattern to match any of the tokens ignoring case
    token_pattern = re.compile("(" + "|".join(sorted_tokens) + ")", flags=re.IGNORECASE)

    result = ""
    i = 0
    while i < len(name):
        m = token_pattern.search(name, i)
        if not m:
            result += name[i:]
            break

        start, end = m.span()
        # Add everything before token
        result += name[i:start]
        token = m.group(0).lower()
        # Add underscores before and after
        if not result.endswith("_") and result and result[-1].isalnum():
            result += "_"
        result += token
        if end < len(name) and name[end].isalnum():
            result += "_"
        i = end

    # Normalize underscores
    result = re.sub(r"__+", "_", result).strip("_")
    return result or name


def camel_to_snake(stem: str) -> str:
    """Standard CamelCase → snake_case conversion."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", stem)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return s.lower()


def to_snake_with_fixed_tokens(stem: str) -> str:
    """Full transformation with token pre-normalization."""
    prepared = pre_normalize_tokens(stem)
    return camel_to_snake(prepared)


def rename_files(base_dir: Path, dry_run: bool = True) -> None:
    """Recursively rename .py files."""
    for path in base_dir.rglob("*.py"):
        if not path.is_file() or path.is_symlink() or path.name == "__init__.py":
            continue

        new_name = to_snake_with_fixed_tokens(path.stem) + path.suffix
        if new_name == path.name:
            continue

        new_path = path.with_name(new_name)
        if new_path.exists():
            print(f"⚠️  Skipping {path.name} → {new_name} (target exists)")
            continue

        print(f"{'[DRY-RUN]' if dry_run else '[RENAME]'} {path.name} → {new_name}")
        if not dry_run:
            path.rename(new_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename CamelCase .py files to snake_case, preserving tokens like it2i/t2i/p2p/etc."
    )
    parser.add_argument(
        "directory", type=Path, help="Root directory to process recursively."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without renaming."
    )
    args = parser.parse_args()

    rename_files(args.directory, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
