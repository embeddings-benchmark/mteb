"""Pinned or local prediction sources for prepared reranking tasks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class FirstStagePredictionSource:
    """A prediction JSON in a Hugging Face dataset, pinned to a full commit SHA.

    Use a local ``str`` or ``Path`` instead when predictions are already on disk.
    Files use MTEB's existing ``{subset: {split: {query: {document: score}}}}``
    format, including the top-level ``mteb_model_meta`` entry.
    """

    repo_id: str
    filename: str
    revision: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
            raise ValueError(
                "First-stage prediction sources require a full commit SHA."
            )

    def download(self) -> Path:
        """Resolve the pinned file through the Hugging Face download cache."""
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                revision=self.revision,
                repo_type="dataset",
            )
        )


def load_first_stage_predictions(
    source: str | Path | FirstStagePredictionSource, prediction_filename: str
) -> dict[str, Any]:
    """Read predictions in the existing MTEB format."""
    path = (
        source.download()
        if isinstance(source, FirstStagePredictionSource)
        else Path(source)
    )
    if path.is_dir():
        path /= prediction_filename
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
