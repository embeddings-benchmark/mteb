"""MTEB registration for the two evaluated text embedding recipes.

Requires an independently configured vLLM-Last-Hidden-State server. Prompt
preflight checks detect template drift; they cannot attest loaded weight bytes.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    import httpx
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class Recipe(TypedDict):
    name: str
    checkpoint: str
    revision: str
    parameters: int
    prompt: str
    chat_template_kwargs: dict[str, str | bool]


RECIPES: dict[str, Recipe] = {
    "qwen_e5v": {
        "name": "Qwen/Qwen3.5-4B-e5v-last-prompt",
        "checkpoint": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "parameters": 4659865088,
        "prompt": "{text}\n Summary above sentence in one word: ",
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "nuextract_revised": {
        "name": "numind/NuExtract3-revised-last-prompt",
        "checkpoint": "numind/NuExtract3",
        "revision": "c99dc8f5641b866aa0192b6ea78f84bf9f3535f1",
        "parameters": 4539265536,
        "prompt": "{text}",
        "chat_template_kwargs": {"mode": "embeddings", "enable_thinking": False},
    },
}


class LastPromptEncoder(AbsEncoder):
    """Text-only adapter preserving the evaluated HTTP request and vector dtype."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        recipe: str,
        base_url: str | None = None,
        served_model: str | None = None,
        concurrency: int = 8,
        timeout: float = 120,
        transport: httpx.BaseTransport | None = None,
        **kwargs: Any,
    ) -> None:
        import httpx

        self.recipe = RECIPES[recipe]
        if model_name != self.recipe["name"] or revision != self.recipe["revision"]:
            raise ValueError("Model name/revision does not match the registered recipe")
        if concurrency < 1 or timeout <= 0:
            raise ValueError("Concurrency and timeout must be positive")
        # MTEB can forward a device; inference runs on the configured server.
        kwargs.pop("device", None)
        if kwargs:
            raise ValueError(f"Unsupported recipe overrides: {sorted(kwargs)}")
        base_url = (
            base_url
            or os.environ.get("LAST_PROMPT_BASE_URL", "http://localhost:8000/v1")
        ).rstrip("/")
        if not base_url.endswith("/v1"):
            raise ValueError("base_url must end in /v1")
        self.model = (
            served_model
            or os.environ.get("LAST_PROMPT_SERVED_MODEL")
            or self.recipe["checkpoint"]
        )
        self.concurrency = concurrency
        headers = {}
        if key := os.environ.get("LAST_PROMPT_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        self.http = httpx.Client(
            base_url=base_url + "/",
            headers=headers,
            timeout=timeout,
            trust_env=False,
            transport=transport,
        )
        try:
            models = self.http.get("models")
            models.raise_for_status()
            info = next(
                (m for m in models.json()["data"] if m["id"] == self.model), None
            )
            if info is None or info.get("root") != self.recipe["checkpoint"]:
                raise ValueError("Server model ID/root does not match the checkpoint")
            check = json.loads(
                (
                    Path(__file__).parent
                    / "last_prompt_resources"
                    / "prompt_checks.json"
                ).read_text()
            )[recipe]
            payload = self.payload(check["text"])
            tokenize = {
                k: payload[k]
                for k in [
                    "model",
                    "messages",
                    "chat_template_kwargs",
                    "add_generation_prompt",
                ]
            }
            response = self.http.post(base_url[:-3] + "/tokenize", json=tokenize)
            response.raise_for_status()
            if response.json()["tokens"] != check["tokens"]:
                raise ValueError(
                    "Effective prompt tokens differ from the evaluated recipe"
                )
        except BaseException:
            self.http.close()
            raise

    def payload(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("Only text inputs are supported")
        return {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.recipe["prompt"].format(text=text)}
            ],
            "max_completion_tokens": 1,
            "temperature": 0,
            "n": 1,
            "stream": False,
            "chat_template_kwargs": dict(self.recipe["chat_template_kwargs"]),
            "add_generation_prompt": True,
            "kv_transfer_params": {"return_last_hidden_state": True},
        }

    def _one(self, text: str) -> np.ndarray:
        response = self.http.post("chat/completions", json=self.payload(text))
        response.raise_for_status()
        vector = np.asarray(
            response.json()["kv_transfer_params"]["last_hidden_state"], dtype=np.float32
        )
        if (
            vector.shape != (2560,)
            or not np.isfinite(vector).all()
            or not np.linalg.norm(vector)
        ):
            raise ValueError("Expected a finite, nonzero 2560-dimensional hidden state")
        return vector

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
        batches = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for batch in inputs:
                if "text" not in batch or {"image", "audio", "video"}.intersection(
                    batch
                ):
                    raise ValueError("This registration accepts text-only batches")
                # Retrieval batches also carry title/body/query/instruction fields.
                # MTEB already combines them into text; use that field unchanged,
                # exactly as the frozen clients used for the benchmark runs did.
                rows = list(pool.map(self._one, batch["text"]))
                if rows:
                    batches.append(np.stack(rows))
        return (
            np.concatenate(batches)
            if batches
            else np.empty((0, 2560), dtype=np.float32)
        )

    def close(self) -> None:
        self.http.close()


def make_meta(recipe: str) -> ModelMeta:
    config = RECIPES[recipe]
    return ModelMeta(
        loader=LastPromptEncoder,
        loader_kwargs={"recipe": recipe},
        name=config["name"],
        revision=config["revision"],
        release_date=None,
        languages=None,
        n_parameters=config["parameters"],
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=2560,
        license="apache-2.0",
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        extra_requirements_groups=["last-prompt"],
        framework=["API"],
        similarity_fn_name="cosine",
        use_instructions=True,
        training_datasets=None,
        modalities=["text"],
        model_type=["dense"],
        reference="https://huggingface.co/" + config["checkpoint"],
        adapted_from=config["checkpoint"],
    )


qwen35_e5v_last_prompt = make_meta("qwen_e5v")
nuextract3_revised_last_prompt = make_meta("nuextract_revised")
