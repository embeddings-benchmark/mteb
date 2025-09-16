from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.requires_package import requires_package
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class MultiVectorModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        """Wrapper for MultiVector/ColBERT models (via PyLate)."""
        requires_package(self, "pylate", model_name, "pip install mteb[pylate]")
        from pylate import models as colbert_model  # type: ignore[import]

        self.model_name = model_name
        self.model = colbert_model.ColBERT(self.model_name, revision=revision, **kwargs)
        built_in_prompts = getattr(self.model, "prompts", None)
        if built_in_prompts and not model_prompts:
            model_prompts = built_in_prompts
        elif model_prompts and built_in_prompts:
            logger.info(f"Model.prompts will be overwritten with {model_prompts}")
            self.model.prompts = self.validate_task_to_prompt_name(model_prompts)

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
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_metadata.name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_metadata.name} prompt_type={prompt_type}"
            )
        logger.debug(f"Encoding {len(inputs)} items.")

        if "request_qid" in kwargs:
            kwargs.pop("request_qid")

        inputs = [text for batch in inputs for text in batch["text"]]

        pred = self.model.encode(
            inputs,
            prompt_name=prompt_name,
            is_query=True if prompt_type == PromptType.query else False,
            convert_to_tensor=True,
            **kwargs,
        )

        # encode returns a list of tensors shaped (x, token_dim), pad to uniform length
        pred = torch.nn.utils.rnn.pad_sequence(pred, batch_first=True, padding_value=0)
        return pred.cpu().numpy()

 

colbert_v2 = ModelMeta(
    loader=MultiVectorModel,
    name="colbert-ir/colbertv2.0",
    languages=["eng-Latn"],
    open_weights=True,
    revision="c1e84128e85ef755c096a95bdb06b47793b13acf",
    public_training_code=None,
    public_training_data=None,
    release_date="2024-09-21",
    n_parameters=int(110 * 1e6),
    memory_usage_mb=418,
    max_tokens=180,
    embed_dim=None,
    license="mit",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["PyLate", "ColBERT"],
    is_pylate_compatible=True,
    reference="https://huggingface.co/colbert-ir/colbertv2.0",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        "MSMARCO",
        "mMARCO-NL",
    },
)

jina_colbert_v2 = ModelMeta(
    loader=MultiVectorModel,
    loader_kwargs=dict(
        query_prefix="[QueryMarker]",
        document_prefix="[DocumentMarker]",
        attend_to_expansion_tokens=True,
        trust_remote_code=True,
    ),
    name="jinaai/jina-colbert-v2",
    languages=[
        "ara-Arab",
        "ben-Beng",
        "deu-Latn",
        "spa-Latn",
        "eng-Latn",
        "fas-Arab",
        "fin-Latn",
        "fra-Latn",
        "hin-Deva",
        "ind-Latn",
        "jpn-Jpan",
        "kor-Kore",
        "rus-Cyrl",
        "swa-Latn",
        "tel-Telu",
        "tha-Thai",
        "yor-Latn",
        "zho-Hans",
        "nld-Latn",
        "ita-Latn",
        "por-Latn",
        "vie-Latn",
    ],
    open_weights=True,
    revision="4cf816e5e2b03167b132a3c847a9ecd48ba708e1",
    public_training_code=None,
    public_training_data=None,
    release_date="2024-08-16",
    n_parameters=int(559 * 1e6),
    memory_usage_mb=1067,
    max_tokens=8192,
    embed_dim=None,
    license="cc-by-nc-4.0",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["PyLate", "ColBERT"],
    is_pylate_compatible=True,
    reference="https://huggingface.co/jinaai/jina-colbert-v2",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        "MSMARCO",
        "mMARCO-NL",
        "DuRetrieval",
        "MIRACL",
    },
)


lightonai__gte_moderncolbert_v1 = ModelMeta(
    loader=MultiVectorModel,
    name="lightonai/GTE-ModernColBERT-v1",
    languages=[
        "eng-Latn",
    ],
    open_weights=True,
    revision="78d50a162b04dfdc45c3af6b4294ba77c24888a3",
    public_training_code="https://gist.github.com/NohTow/3030fe16933d8276dd5b3e9877d89f0f",
    public_training_data="https://huggingface.co/datasets/lightonai/ms-marco-en-bge-gemma",
    release_date="2025-04-30",
    n_parameters=int(149 * 1e6),
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=None,
    license="apache-2.0",
    similarity_fn_name="MaxSim",
    framework=["PyLate", "ColBERT"],
    is_pylate_compatible=True,
    reference="https://huggingface.co/lightonai/GTE-ModernColBERT-v1",
    use_instructions=False,
    adapted_from="Alibaba-NLP/gte-modernbert-base",
    superseded_by=None,
    training_datasets={
        "MSMARCO",
        "mMARCO-NL",
    },
)


# Additional PyLate-compatible ColBERT model(s)
lightonai__answerai_colbert_small_v1 = ModelMeta(
    loader=MultiVectorModel,
    name="lightonai/answerai-colbert-small-v1",
    languages=[
        "eng-Latn",
    ],
    open_weights=True,
    revision=None,
    public_training_code=None,
    public_training_data=None,
    release_date=None,
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["PyLate", "ColBERT"],
    reference=None,
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets=None,
    is_pylate_compatible=True,
)
