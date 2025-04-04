from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import torch

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class ColBERTWrapper(Wrapper):
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
        from pylate import models as colbert_model

        self.model_name = model_name
        self.model = colbert_model.ColBERT(self.model_name, revision=revision, **kwargs)
        if (
            model_prompts is None
            and hasattr(self.model, "prompts")
            and len(self.model.prompts) > 0
        ):
            try:
                model_prompts = self.validate_task_to_prompt_name(self.model.prompts)
            except ValueError:
                model_prompts = None
        elif model_prompts is not None and hasattr(self.model, "prompts"):
            logger.info(f"Model prompts will be overwritten with {model_prompts}")
            self.model.prompts = model_prompts
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task. Pylate uses this to
                determine which prompt to use from a specified dictionary.
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the encoder.

            The order of priorities for prompt selection are:
                1. Composed prompt of task name + prompt type (query or passage)
                2. Specific task prompt
                3. Composed prompt of task type + prompt type (query or passage)
                4. Specific task type prompt
                5. Specific prompt type (query or passage)

        Returns:
            The encoded sentences as a numpy array.
        """
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = self.get_prompt_name(
                self.model_prompts, task_name, prompt_type
            )
        if prompt_name:
            logger.info(
                f"Using prompt_name={prompt_name} for task={task_name} prompt_type={prompt_type}"
            )
        else:
            logger.info(
                f"No model prompts found for task={task_name} prompt_type={prompt_type}"
            )
        logger.info(f"Encoding {len(sentences)} sentences.")

        pred = self.model.encode(
            sentences,
            prompt_name=prompt_name,
            is_query=True if prompt_type == PromptType.query else False,
            **kwargs,
        )

        # encode returns a list of tensors shaped (x, token_dim) where x is the number of tokens in the sentence
        # we need to pad these tensors to the same length
        # Tensors have varying lengths; therefore, they need to be padded with zeros to ensure uniformity before being combined
        # output shape will be (batch_size, len(max(tokens)), embedding_token_dim)
        pred = torch.nn.utils.rnn.pad_sequence(pred, batch_first=True, padding_value=0)

        return pred.cpu().numpy()

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Computes the max-similarity max_sim(a[i], b[j]) for all i and j.
            Works with a Tensor of the shape (batch_size, num_tokens, token_dim)

        Return:
            Matrix with res[i][j]  = max_sim(a[i], b[j])
        """  # noqa: D402
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32)

        if len(a.shape) == 2:
            a = a.unsqueeze(0)

        if len(b.shape) == 2:
            b = b.unsqueeze(0)

        scores = torch.einsum(
            "ash,bth->abst",
            a,
            b,
        )

        return scores.max(axis=-1).values.sum(axis=-1)


colbert_v2 = ModelMeta(
    loader=partial(
        ColBERTWrapper,
        model_name="colbert-ir/colbertv2.0",
    ),
    name="colbert-ir/colbertv2.0",
    languages=["eng_Latn"],
    open_weights=True,
    revision="c1e84128e85ef755c096a95bdb06b47793b13acf",
    public_training_code=None,
    public_training_data=None,
    release_date="2024-09-21",
    n_parameters=110 * 1e6,
    memory_usage_mb=418,
    max_tokens=180,  # Reduced for Benchmarking - see ColBERT paper
    embed_dim=None,  # Bag of Embeddings (128) for each token
    license="mit",
    similarity_fn_name="max_sim",
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

jina_colbert_v2 = ModelMeta(
    loader=partial(
        ColBERTWrapper,
        model_name="jinaai/jina-colbert-v2",
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
    n_parameters=559 * 1e6,
    memory_usage_mb=1067,
    max_tokens=8192,
    embed_dim=None,  # Bag of Embeddings (128) for each token
    license="cc-by-nc-4.0",
    similarity_fn_name="max_sim",
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
