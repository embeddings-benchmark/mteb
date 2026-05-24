"""MTEB API loader for jcorners/ingot-8b-r3 — served at https://api-mteb.voxell.ai/.

To enable: drop this file at mteb/models/model_implementations/ingot_models.py in
the embeddings-benchmark/mteb fork. mteb 2.12.30 auto-discovers files in that
directory; no edit to mteb/models/__init__.py is needed.

Environment:
    MTEB_API_KEY  Bearer token. Required — no demo key is shipped in this file.
                  Request a per-reviewer key via https://voxell.ai (reference
                  your MTEB PR/eval); 200 rpm slot, audited per tenant.
    MTEB_API_URL  Override the base URL (default https://api-mteb.voxell.ai).
"""
from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import requests

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

MTEB_API_DEMO_KEY = None  # no key shipped in this file; reviewers request per-reviewer keys (see module docstring)
DEFAULT_BASE_URL = "https://api-mteb.voxell.ai"
# Client-side body cap. Gateway enforces 8 MiB (see gateway/mteb-api-gateway/src/index.ts:35);
# 7 MiB default leaves ~1 MiB headroom for headers + JSON overhead.
MAX_BODY_MB = int(os.environ.get("MTEB_API_MAX_BODY_MB", "7"))


class IngotAPIEncoder(AbsEncoder):
    def __init__(
        self,
        model_name: str = "jcorners/ingot-8b-r3",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        max_retries: int = 3,
        **_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.base_url = (base_url or os.environ.get("MTEB_API_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("MTEB_API_KEY") or MTEB_API_DEMO_KEY
        if self.api_key is None:
            raise RuntimeError(
                "MTEB_API_KEY not set and no demo key shipped. "
                "Request a per-reviewer key via https://voxell.ai (reference your MTEB PR/eval); "
                "200 rpm slot, audited per tenant."
            )
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._session = requests.Session()
        # Identity encoding: float-vector responses don't compress meaningfully and
        # we saw brotli decode errors on large STS responses through Cloudflare
        # (Inc-4 C3 smoke). Skip compression altogether.
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept-Encoding": "identity",
        })

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **_kwargs: Any,
    ) -> Array:
        # Single POST per task: Qwen3-8B uses left-padding + last-token pooling,
        # which is NOT pool-invariant. Sub-batching here would produce different
        # per-sentence vectors than the in-process eval (STS12 86.40 -> 81.41
        # measured under client-side batch_size=64 chunking).
        sentences = [text for batch in inputs for text in batch["text"]]
        payload: dict[str, Any] = {
            "input": sentences,
            "model": self.model_name,
            "task_name": task_metadata.name,
        }
        if prompt_type is not None:
            payload["prompt_type"] = prompt_type.value

        body_bytes = len(json.dumps(payload).encode("utf-8"))
        max_bytes = MAX_BODY_MB * 1024 * 1024
        if body_bytes > max_bytes:
            raise RuntimeError(
                f"ingot api body too large ({body_bytes} bytes, {len(sentences)} sentences); "
                f"MTEB_API_MAX_BODY_MB={MAX_BODY_MB}. Raise it or split the task upstream."
            )

        data = self._post_with_retry(f"{self.base_url}/v1/embeddings", payload)
        return np.asarray([item["embedding"] for item in data["data"]], dtype=np.float32)

    def _post_with_retry(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        backoff = 1.5
        last_5xx_body = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session.post(url, json=payload, timeout=self.timeout_s)
                status = r.status_code
                if status in (401, 403):
                    raise RuntimeError(
                        f"ingot api auth failed (status {status}): "
                        f"MTEB_API_KEY rejected. Request a per-reviewer key via https://voxell.ai."
                    )
                if status == 413:
                    raise RuntimeError(
                        f"ingot api rejected body (status 413): "
                        f"server limit exceeded; raise MTEB_API_MAX_BODY_MB or shrink task"
                    )
                if status == 429:
                    retry_after = float(r.headers.get("Retry-After", backoff))
                    time.sleep(retry_after)
                    continue
                if 500 <= status < 600:
                    last_5xx_body = r.text[:500]
                    if attempt == self.max_retries:
                        raise RuntimeError(
                            f"ingot api {url} 5xx ({status}) after {attempt} attempts: {last_5xx_body}"
                        )
                    time.sleep(backoff * attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"ingot api {url} unreachable after {attempt} attempts: {e}"
                    ) from e
                time.sleep(backoff * attempt)
        raise RuntimeError(f"ingot api {url} exceeded {self.max_retries} retries")


ingot_8b_r3 = ModelMeta(
    name="jcorners/ingot-8b-r3",
    revision="r3",
    release_date="2026-05-23",
    languages=["eng-Latn"],
    loader=IngotAPIEncoder,
    loader_kwargs=dict(),
    max_tokens=32768,
    embed_dim=4096,
    open_weights=False,
    n_parameters=8_000_000_000,
    n_embedding_parameters=8_000_000_000,
    memory_usage_mb=None,
    license=None,
    reference="https://voxell.ai/engineering/ingot_poured/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
        "MedrxivClusteringP2P.v2",
        "MedrxivClusteringS2S.v2",
    },
)
