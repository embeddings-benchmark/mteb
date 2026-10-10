"""Named, reproducible sources for the existing two-stage reranking workflow."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
) -> tuple[dict[str, Any], str]:
    """Read and fingerprint exactly the bytes used for conversion."""
    path = (
        source.download()
        if isinstance(source, FirstStagePredictionSource)
        else Path(source)
    )
    if path.is_dir():
        path /= prediction_filename
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()
