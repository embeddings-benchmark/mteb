"""ModelMeta for Ingot-8B-R3 (JCorners/Ingot-8B-R3).

Ingot-8B-R3 is built on Qwen3-Embedding-8B with a proprietary routing
framework. Different specialists activate at inference time from input
content alone (no task metadata required). The routing framework is
patent-pending.

Access: API — no weights download required. The model is served through
the Voxell Forge API at api.voxell.ai. An API key is required; request
access at https://huggingface.co/JCorners/Ingot-8B-R3 or contact
corp@voxell.ai. Set the VOXELL_API_KEY environment variable to use this
loader.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

logger = logging.getLogger(__name__)

_FORGE_BASE_URL = "https://api.voxell.ai/v1"
_MODEL_ID = "JCorners/Ingot-8B-R3"


def _instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = next(iter(instruction.values()))
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery: "


class IngotForgeModel(AbsEncoder):
    """Calls the Voxell Forge API for Ingot-8B-R3 embeddings.

    Set VOXELL_API_KEY in the environment. Requests use the OpenAI-compatible
    /v1/embeddings endpoint at api.voxell.ai.
    """

    def __init__(self, model_name: str = _MODEL_ID, **kwargs: Any) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai>=1.0 is required for Ingot-8B-R3. Install with: pip install openai"
            ) from e

        api_key = os.environ.get("VOXELL_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "VOXELL_API_KEY environment variable is not set. "
                "Request access at https://huggingface.co/JCorners/Ingot-8B-R3 "
                "or contact corp@voxell.ai."
            )
        self._client = OpenAI(api_key=api_key, base_url=_FORGE_BASE_URL)
        self._model_name = model_name

    def _embed_batch(
        self, texts: list[str], input_type: str | None = None
    ) -> np.ndarray:
        extra = {}
        if input_type:
            extra["extra_body"] = {"input_type": input_type}
        response = self._client.embeddings.create(
            model=self._model_name,
            input=texts,
            **extra,
        )
        return np.array([e.embedding for e in response.data], dtype=np.float32)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        input_type = None
        if prompt_type == PromptType.query:
            input_type = "query"
        elif prompt_type == PromptType.document:
            input_type = "passage"

        all_embeddings: list[np.ndarray] = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Ingot Forge"):
            texts = batch["text"]
            # Apply instruction prefix for query-type inputs.
            if input_type == "query" and task_metadata is not None:
                instruction = getattr(task_metadata, "prompt", None)
                if instruction:
                    prefix = _instruction_template(instruction, prompt_type)
                    texts = [prefix + t for t in texts]
            embs = self._embed_batch(texts, input_type=input_type)
            all_embeddings.append(embs)
        return np.vstack(all_embeddings)


def ingot_8b_r3_loader(model_name_or_path: str, **kwargs: Any) -> IngotForgeModel:
    return IngotForgeModel(model_name=model_name_or_path, **kwargs)


Ingot_8B_R3 = ModelMeta(
    loader=ingot_8b_r3_loader,
    name="JCorners/Ingot-8B-R3",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=False,
    revision="1",
    release_date="2026-05-21",
    n_parameters=7_567_295_488,
    n_embedding_parameters=621_219_840,
    memory_usage_mb=None,
    embed_dim=4096,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/JCorners/Ingot-8B-R3",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        # Inherited from Qwen3-Embedding-8B base model
        "T2Retrieval",
        "DuRetrieval",
        "MMarcoReranking",
        "CMedQAv2-reranking",
        "NQ",
        "MSMARCO",
        "HotpotQA",
        "FEVER",
        "MrTidyRetrieval",
        "MIRACLRetrieval",
        "CodeSearchNet",
    },
)
