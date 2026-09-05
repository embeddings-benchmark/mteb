#!/usr/bin/env python3
"""MOEB Contamination Report.

Generates a contamination matrix showing which MOEB models (those with
audio/video/image modalities) have training data that overlaps with MOEB
evaluation tasks.

Three-valued contamination status per model:
  CLEAN    -- training_datasets is set and contains no MOEB task names
  UNKNOWN  -- training_datasets is None (author has not declared training data)
  CONTAMINATED -- training_datasets overlaps with one or more MOEB task names

Usage:
  uv run python scripts/moeb_contamination_report.py
  uv run python scripts/moeb_contamination_report.py --format csv > report.csv
  uv run python scripts/moeb_contamination_report.py --only-unknown
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from typing import Literal

import mteb
from mteb.models.model_meta import ModelMeta

_MOEB_MODALITIES = {"audio", "video", "image"}

Status = Literal["CLEAN", "UNKNOWN", "CONTAMINATED"]


def _is_moeb_model(meta: ModelMeta) -> bool:
    """True if the model handles at least one non-text MOEB modality."""
    return bool(set(meta.modalities or []) & _MOEB_MODALITIES)


def _get_moeb_tasks() -> list:
    """All tasks whose modalities include at least one of audio/video/image."""
    all_tasks = mteb.get_tasks()
    return [
        t
        for t in all_tasks
        if set(t.metadata.modalities or []) & _MOEB_MODALITIES
    ]


def _contamination_status(
    meta: ModelMeta, task_names: set[str]
) -> tuple[Status, set[str]]:
    """Return (status, overlapping_task_names)."""
    if meta.training_datasets is None:
        return "UNKNOWN", set()

    training = meta.get_training_datasets()
    if training is None:
        return "UNKNOWN", set()

    overlap = training & task_names
    if overlap:
        return "CONTAMINATED", overlap
    return "CLEAN", set()


def build_report(
    only_unknown: bool = False,
) -> list[dict]:
    """Return a list of row dicts for all MOEB models."""
    print("Loading MOEB tasks...", file=sys.stderr)
    moeb_tasks = _get_moeb_tasks()
    task_names = {t.metadata.name for t in moeb_tasks}
    print(f"  {len(moeb_tasks)} MOEB tasks found.", file=sys.stderr)

    print("Loading model registry...", file=sys.stderr)
    all_models = mteb.get_model_metas()
    moeb_models = [m for m in all_models if _is_moeb_model(m)]
    print(f"  {len(moeb_models)} MOEB-relevant models found.", file=sys.stderr)

    rows = []
    status_counts: dict[Status, int] = defaultdict(int)

    for meta in sorted(moeb_models, key=lambda m: m.name or ""):
        status, overlap = _contamination_status(meta, task_names)
        status_counts[status] += 1

        if only_unknown and status != "UNKNOWN":
            continue

        rows.append(
            {
                "model": meta.name or "?",
                "modalities": ",".join(sorted(meta.modalities or [])),
                "status": status,
                "contaminated_tasks": "; ".join(sorted(overlap)) if overlap else "",
                "training_datasets_declared": str(meta.training_datasets is not None),
            }
        )

    print(
        f"\nSummary: {status_counts['CLEAN']} CLEAN | "
        f"{status_counts['UNKNOWN']} UNKNOWN | "
        f"{status_counts['CONTAMINATED']} CONTAMINATED",
        file=sys.stderr,
    )
    return rows


def print_markdown(rows: list[dict]) -> None:
    if not rows:
        print("No rows to display.")
        return
    headers = ["Model", "Modalities", "Status", "Contaminated Tasks"]
    col_widths = [
        max(len(h), max((len(r[k]) for r in rows), default=0))
        for h, k in zip(
            headers,
            ["model", "modalities", "status", "contaminated_tasks"],
        )
    ]
    sep = " | ".join("-" * w for w in col_widths)
    header_row = " | ".join(
        h.ljust(w) for h, w in zip(headers, col_widths)
    )
    print(f"| {header_row} |")
    print(f"| {sep} |")
    for r in rows:
        cells = [
            r["model"].ljust(col_widths[0]),
            r["modalities"].ljust(col_widths[1]),
            r["status"].ljust(col_widths[2]),
            r["contaminated_tasks"].ljust(col_widths[3]),
        ]
        print("| " + " | ".join(cells) + " |")


def print_csv(rows: list[dict]) -> None:
    if not rows:
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=["markdown", "csv"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--only-unknown",
        action="store_true",
        help="Only show models with unknown training data (need to be filled in)",
    )
    args = parser.parse_args()

    rows = build_report(only_unknown=args.only_unknown)

    if args.format == "csv":
        print_csv(rows)
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
