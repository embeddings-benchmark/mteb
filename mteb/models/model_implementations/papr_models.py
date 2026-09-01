from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

# Band schema per task. This selects the 14 frequency bands the encoder
# conditions its export gate on, and nothing else -- the query instruction is a
# separate request field that mteb fills from the task metadata. The two
# -scifact models pin their schema server-side and ignore this; the general
# models have no default and require one.
TASK_SCHEMA_IDS: dict[str, str] = {
    "SciFact": "biomedical:scifact:2.0.0",
}

# MTEB tasks whose TRAIN split is in the papr-embed-v1 fine-tune mix.
# Declared so the leaderboard marks these as in-domain. No test split is
# trained on. See https://github.com/Papr-ai/papr-embed-evals
PAPR_TRAINING_DATASETS = {
    "SciFact",
    "NFCorpus",
    "NQ",
    "HotpotQA",
    "FEVER",
    "FiQA2018",
}

DEFAULT_BASE_URL = "https://memory.papr.ai"
MAX_BATCH_SIZE = 64
MAX_INPUT_CHARS = 90_000


class PaprEmbedAPIModel(AbsEncoder):
    """Closed-weights API encoder for papr-embed-v1.

    Auth: ``PAPR_API_KEY`` (https://dashboard.papr.ai). Optional
    ``PAPR_BASE_URL`` overrides the default production endpoint.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        *,
        papr_model_id: str,
        embed_dim: int,
        reasoning: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        api_key = os.getenv("PAPR_API_KEY", "")
        if not api_key:
            raise ValueError(
                "PAPR_API_KEY is not set. Create a key at https://dashboard.papr.ai "
                "(Settings -> API Keys) and export it before running."
            )
        self.model_name = model_name
        self.papr_model_id = papr_model_id
        self._embed_dim = embed_dim
        self.reasoning = reasoning
        self.base_url = os.getenv("PAPR_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self._headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        is_query = _is_query(prompt_type)
        input_type: Literal["query", "document"] = "query" if is_query else "document"
        texts = _collect_texts(inputs)
        if not texts:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        # mteb owns the instruction. The API takes it as an explicit field and
        # applies exactly what it is given, so the prompt this model is scored
        # with is the task's registered one, visible here rather than chosen
        # server-side. Documents are not instructed.
        instruction = (
            self.get_instruction(task_metadata, prompt_type) if is_query else None
        )

        schema_id = TASK_SCHEMA_IDS.get(getattr(task_metadata, "name", "") or "")
        vectors: list[list[float]] = []
        for batch in _batches(texts):
            payload: dict[str, Any] = {
                "model": self.papr_model_id,
                "input": batch,
                "input_type": input_type,
            }
            if is_query:
                payload["instruction"] = instruction or ""
            if schema_id:
                payload["schema_id"] = schema_id
            if self.reasoning:
                payload["reasoning"] = self.reasoning
            body = self._post("/v1/embeddings", payload)
            rows = sorted(body["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in rows)
        return np.asarray(vectors, dtype=np.float32)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        last_error = "unknown error"
        for attempt in range(1, 7):
            request = urllib.request.Request(
                f"{self.base_url}{path}",
                data=data,
                headers=self._headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:300]
                last_error = f"HTTP {exc.code}: {body}"
                if 400 <= exc.code < 500 and exc.code != 429:
                    raise RuntimeError(f"Papr embeddings API: {last_error}") from exc
            except urllib.error.URLError as exc:
                last_error = f"transport error: {exc}"
            time.sleep(min(60.0, 2.0**attempt))
        raise RuntimeError(f"Papr embeddings API failed after retries ({last_error})")


def _is_query(prompt_type: PromptType | None) -> bool:
    if prompt_type is None:
        return False
    value = getattr(prompt_type, "value", prompt_type)
    return str(value).lower() == "query"


def _collect_texts(inputs: DataLoader[BatchedInput]) -> list[str]:
    texts: list[str] = []
    for batch in inputs:
        texts.extend(str(text) for text in batch["text"])
    return texts


def _batches(texts: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for text in texts:
        if current and (
            len(current) >= MAX_BATCH_SIZE or chars + len(text) > MAX_INPUT_CHARS
        ):
            batches.append(current)
            current, chars = [], 0
        current.append(text)
        chars += len(text)
    if current:
        batches.append(current)
    return batches


papr_embed_v1_0_6b_scifact = ModelMeta(
    name="papr/papr-embed-v1-0.6b-scifact",
    model_type=["dense"],
    revision="1",
    release_date="2026-08-20",
    languages=["eng-Latn"],
    loader=PaprEmbedAPIModel,
    loader_kwargs=dict(
        papr_model_id="papr-embed-v1-0.6b-scifact",
        embed_dim=3968,
    ),
    max_tokens=2048,
    embed_dim=3968,
    open_weights=False,
    n_parameters=628_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://github.com/Papr-ai/papr-embed-evals",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=PAPR_TRAINING_DATASETS,
)

papr_embed_v1_0_6b_scifact_reasoning = ModelMeta(
    name="papr/papr-embed-v1-0.6b-scifact-reasoning-frontier-max",
    model_type=["dense"],
    revision="1",
    release_date="2026-08-20",
    languages=["eng-Latn"],
    loader=PaprEmbedAPIModel,
    loader_kwargs=dict(
        papr_model_id="papr-embed-v1-0.6b-scifact",
        embed_dim=3968,
        reasoning={"tier": "frontier", "effort": "max"},
    ),
    max_tokens=2048,
    embed_dim=3968,
    open_weights=False,
    n_parameters=628_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://github.com/Papr-ai/papr-embed-evals",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=PAPR_TRAINING_DATASETS,
)

papr_embed_v1_0_6b = ModelMeta(
    name="papr/papr-embed-v1-0.6b",
    model_type=["dense"],
    revision="1",
    release_date="2026-08-20",
    languages=["eng-Latn"],
    loader=PaprEmbedAPIModel,
    loader_kwargs=dict(
        papr_model_id="papr-embed-v1-0.6b",
        embed_dim=3968,
    ),
    max_tokens=2048,
    embed_dim=3968,
    open_weights=False,
    n_parameters=628_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://github.com/Papr-ai/papr-embed-evals",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=PAPR_TRAINING_DATASETS,
)

papr_embed_v1_0_6b_reasoning = ModelMeta(
    name="papr/papr-embed-v1-0.6b-reasoning-frontier-max",
    model_type=["dense"],
    revision="1",
    release_date="2026-08-20",
    languages=["eng-Latn"],
    loader=PaprEmbedAPIModel,
    loader_kwargs=dict(
        papr_model_id="papr-embed-v1-0.6b",
        embed_dim=3968,
        reasoning={"tier": "frontier", "effort": "max"},
    ),
    max_tokens=2048,
    embed_dim=3968,
    open_weights=False,
    n_parameters=628_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://github.com/Papr-ai/papr-embed-evals",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=PAPR_TRAINING_DATASETS,
)

papr_embed_v1_4b = ModelMeta(
    name="papr/papr-embed-v1-4b",
    model_type=["dense"],
    revision="1",
    release_date="2026-08-20",
    languages=["eng-Latn"],
    loader=PaprEmbedAPIModel,
    loader_kwargs=dict(
        papr_model_id="papr-embed-v1-4b",
        embed_dim=5504,
    ),
    max_tokens=2048,
    embed_dim=5504,
    open_weights=False,
    n_parameters=4_020_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://github.com/Papr-ai/papr-embed-evals",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=PAPR_TRAINING_DATASETS,
)

papr_embed_v1_4b_reasoning = ModelMeta(
    name="papr/papr-embed-v1-4b-reasoning-frontier-max",
    model_type=["dense"],
    revision="1",
    release_date="2026-08-20",
    languages=["eng-Latn"],
    loader=PaprEmbedAPIModel,
    loader_kwargs=dict(
        papr_model_id="papr-embed-v1-4b",
        embed_dim=5504,
        reasoning={"tier": "frontier", "effort": "max"},
    ),
    max_tokens=2048,
    embed_dim=5504,
    open_weights=False,
    n_parameters=4_020_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://github.com/Papr-ai/papr-embed-evals",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=PAPR_TRAINING_DATASETS,
)
