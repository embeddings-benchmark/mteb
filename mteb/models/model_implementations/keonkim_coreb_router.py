from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations.codefuse_models import (
    C2LLM_7B,
    F2LLM_v2_330M,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Route:
    model_name: str
    revision: str


ROUTES = {
    "Retrieval": Route(
        model_name=F2LLM_v2_330M.name,
        revision=F2LLM_v2_330M.revision,
    ),
    "Reranking": Route(
        model_name=C2LLM_7B.name,
        revision=C2LLM_7B.revision,
    ),
}

ModelLoader = Callable[[Route], Any]


class CoREBTaskTypeRouter(AbsEncoder):
    """Route encodes using only the coarse MTEB task type.

    F2LLM-v2-330M handles Retrieval and C2LLM-7B handles Reranking. Only one
    child remains resident when ``release_between_types`` is enabled.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        *,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        release_between_types: bool = True,
        model_loader: ModelLoader | None = None,
        **_: Any,
    ) -> None:
        del model_name, revision
        self.device = device
        self.model_kwargs = model_kwargs or {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.bfloat16,
        }
        self.release_between_types = release_between_types
        self._injected_loader = model_loader
        self._models: dict[str, Any] = {}
        self._active_task_type: str | None = None

    def _load_model(self, task_type: str) -> Any:
        if task_type in self._models:
            return self._models[task_type]

        if self.release_between_types and self._active_task_type != task_type:
            self.release()

        route = ROUTES[task_type]
        logger.info(
            "Routing MTEB task type %s to %s at %s",
            task_type,
            route.model_name,
            route.revision,
        )
        if self._injected_loader is not None:
            model = self._injected_loader(route)
        else:
            model = mteb.get_model(
                route.model_name,
                revision=route.revision,
                device=self.device,
                model_kwargs=self.model_kwargs,
            )

        self._models[task_type] = model
        self._active_task_type = task_type
        return model

    def release(self) -> None:
        """Release child references and unused CUDA caching allocations."""

        self._models.clear()
        self._active_task_type = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        task_type = task_metadata.type
        if task_type not in ROUTES:
            supported = ", ".join(sorted(ROUTES))
            raise ValueError(
                f"Unsupported task type {task_type!r}; expected one of {supported}"
            )

        model = self._load_model(task_type)
        return model.encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )


_CHILD_METAS = (F2LLM_v2_330M, C2LLM_7B)
_TRAINING_DATASETS = set().union(
    *(meta.training_datasets or set() for meta in _CHILD_METAS)
)
_LANGUAGES = sorted(set().union(*(set(meta.languages or []) for meta in _CHILD_METAS)))
_FRAMEWORKS = sorted(set().union(*(set(meta.framework or []) for meta in _CHILD_METAS)))


coreb_task_type_router = ModelMeta(
    loader=CoREBTaskTypeRouter,
    loader_kwargs={"release_between_types": True},
    name="keonkim/coreb-task-type-router-f2llmv2-330m-c2llm-7b",
    # Replace with the Hugging Face model-card commit before submitting.
    revision="TODO_AFTER_HF_UPLOAD",
    release_date="2026-07-26",
    languages=_LANGUAGES,
    n_parameters=sum(meta.n_parameters or 0 for meta in _CHILD_METAS),
    n_embedding_parameters=sum(
        meta.n_embedding_parameters or 0 for meta in _CHILD_METAS
    ),
    # Children are loaded one at a time, so peak weight memory is the larger
    # child's footprint rather than the sum.
    memory_usage_mb=max(meta.memory_usage_mb or 0 for meta in _CHILD_METAS),
    # The public wrappers currently configure 8,192 tokens for Retrieval and
    # 2,048 for Reranking; this is the guaranteed common maximum.
    max_tokens=2048,
    # This task router emits 896d Retrieval and 3584d Reranking embeddings.
    # ModelMeta currently has no task-dependent dimension representation.
    embed_dim=None,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=_FRAMEWORKS,
    reference="https://huggingface.co/keonkim/coreb-task-type-router-f2llmv2-330m-c2llm-7b",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=_TRAINING_DATASETS,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["router", "dense"],
    citation=f"{F2LLM_v2_330M.citation}\n{C2LLM_7B.citation}",
    contacts=None,
)
