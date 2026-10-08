"""Named, reproducible sources for the existing two-stage reranking workflow."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field


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
    document_representation: Literal["text", "image", "text-image"] | None = None

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


class PredictionArtifact(BaseModel):
    """The exact prediction bytes used in an evaluation."""

    sha256: str
    repo_id: str | None = None
    revision: str | None = None
    filename: str | None = None


class RerankingConfiguration(BaseModel):
    """First-stage evaluation context, independent of reranker model settings."""

    first_stage: str
    top_k: int = Field(gt=0)
    document_representation: Literal["text", "image", "text-image"] | None = None
    predictions: PredictionArtifact

    @property
    def configuration_id(self) -> str:
        """Group domains from one pinned prediction collection and candidate depth.

        Local files instead use their content hash. A pinned Hub directory defines
        a collection; filenames and checksums remain per-task provenance and are
        checked when reusing or merging that task's results.
        """
        identity = self.model_dump()
        if self.predictions.repo_id is not None:
            if self.predictions.filename is None or self.predictions.revision is None:
                raise ValueError(
                    "Hub prediction provenance requires a filename and revision."
                )
            identity["predictions"] = {
                "repo_id": self.predictions.repo_id,
                "revision": self.predictions.revision,
                "directory": str(PurePosixPath(self.predictions.filename).parent),
            }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        return f"cfg_{digest[:16]}"
