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


class ColBERTModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        """Wrapper for ColBERT models.

        Args:
            model_name: The ColBERT model to load from HuggingFace Hub.
            revision: The revision of the model to use.
            model_prompts: A dictionary mapping task names to prompt names.
                First priority is given to the composed prompt of task name + prompt type (query or passage), then to the specific task prompt,
                then to the composed prompt of task type + prompt type, then to the specific task type prompt,
                and finally to the specific prompt type.
            **kwargs: Additional arguments to pass to the model.
        """
        requires_package(self, "pylate", model_name, "pip install mteb[pylate]")
        from pylate import models as colbert_model  # type: ignore[import]

        self.model_name = model_name
        self.model = colbert_model.ColBERT(self.model_name, revision=revision, **kwargs)
        if (
            model_prompts is None
            and hasattr(self.model, "prompts")
            and len(self.model.prompts) > 0
        ):
            try:
                self.model_prompts = model_prompts
                self.validate_task_to_prompt_name()
            except ValueError:
                model_prompts = None
        elif model_prompts is not None and hasattr(self.model, "prompts"):
            logger.info(f"Model prompts will be overwritten with {model_prompts}")
            self.model.prompts = model_prompts
            self.validate_task_to_prompt_name()

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
        logger.info(f"Encoding {len(inputs)} sentences.")

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

        # encode returns a list of tensors shaped (x, token_dim) where x is the number of tokens in the sentence
        # we need to pad these tensors to the same length
        # Tensors have varying lengths; therefore, they need to be padded with zeros to ensure uniformity before being combined
        # output shape will be (batch_size, len(max(tokens)), embedding_token_dim)
        pred = torch.nn.utils.rnn.pad_sequence(pred, batch_first=True, padding_value=0)

        return pred.cpu().numpy()


colbert_v2 = ModelMeta.model_construct(
    loader=ColBERTModel,
    name="colbert-ir/colbertv2.0",
    languages=["eng-Latn"],
    open_weights=True,
    revision="c1e84128e85ef755c096a95bdb06b47793b13acf",
    public_training_code=None,
    public_training_data=None,
    release_date="2024-09-21",
    n_parameters=int(110 * 1e6),
    memory_usage_mb=418,
    max_tokens=180,  # Reduced for Benchmarking - see ColBERT paper
    embed_dim=None,  # Bag of Embeddings (128) for each token
    license="mit",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["PyLate", "ColBERT"],
    reference="https://huggingface.co/colbert-ir/colbertv2.0",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        "MSMARCO": ["train"],  # dev?
        "mMARCO-NL": ["train"],  # translation not trained on
    },
)

jina_colbert_v2 = ModelMeta.model_construct(
    loader=ColBERTModel,
    loader_kwargs=dict(
        query_prefix="[QueryMarker]",
        document_prefix="[DocumentMarker]",
        attend_to_expansion_tokens=True,
        trust_remote_code=True,
    ),
    name="jinaai/jina-colbert-v2",
    languages=[  # list of languages the model has been evaluated on
        "ara-Arab",  # Arabic
        "ben-Beng",  # Bengali
        "deu-Latn",  # German
        "spa-Latn",  # Spanish
        "eng-Latn",  # English
        "fas-Arab",  # Persian
        "fin-Latn",  # Finnish
        "fra-Latn",  # French
        "hin-Deva",  # Hindi
        "ind-Latn",  # Indonesian
        "jpn-Jpan",  # Japanese
        "kor-Kore",  # Korean
        "rus-Cyrl",  # Russian
        "swa-Latn",  # Swahili
        "tel-Telu",  # Telugu
        "tha-Thai",  # Thai
        "yor-Latn",  # Yoruba
        "zho-Hans",  # Chinese (Simplified)
        "nld-Latn",  # Dutch
        "ita-Latn",  # Italian
        "por-Latn",  # Portuguese
        "vie-Latn",  # Vietnamese
    ],
    open_weights=True,
    revision="4cf816e5e2b03167b132a3c847a9ecd48ba708e1",
    public_training_code=None,
    public_training_data=None,
    release_date="2024-08-16",
    n_parameters=int(559 * 1e6),
    memory_usage_mb=1067,
    max_tokens=8192,
    embed_dim=None,  # Bag of Embeddings (128) for each token
    license="cc-by-nc-4.0",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["PyLate", "ColBERT"],
    reference="https://huggingface.co/jinaai/jina-colbert-v2",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        "MSMARCO": ["train"],
        "mMARCO-NL": ["train"],  # translation not trained on
        "DuRetrieval": [],
        "MIRACL": ["train"],
    },
)


lightonai__GTE_ModernColBERT_v1 = ModelMeta.model_construct(
    loader=ColBERTModel,
    name="lightonai/GTE-ModernColBERT-v1",
    languages=[
        "eng-Latn",  # English
    ],
    open_weights=True,
    revision="78d50a162b04dfdc45c3af6b4294ba77c24888a3",
    public_training_code="https://gist.github.com/NohTow/3030fe16933d8276dd5b3e9877d89f0f",
    public_training_data="https://huggingface.co/datasets/lightonai/ms-marco-en-bge-gemma",
    release_date="2025-04-30",
    n_parameters=int(149 * 1e6),
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=None,  # Bag of Embeddings (128) for each token
    license="apache-2.0",
    similarity_fn_name="MaxSim",
    framework=["PyLate", "ColBERT"],
    reference="https://huggingface.co/lightonai/GTE-ModernColBERT-v1",
    use_instructions=False,
    adapted_from="Alibaba-NLP/gte-modernbert-base",
    superseded_by=None,
    training_datasets={
        "MSMARCO": ["train"],
        "mMARCO-NL": ["train"],  # translation not trained on
    },
)
