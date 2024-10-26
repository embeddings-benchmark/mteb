from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from mteb.encoder_interface import Encoder, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import (
    get_prompt_name,
    validate_task_to_prompt_name,
)

from .wrapper import Wrapper


class GoogleTextEmbeddingModel(Encoder, Wrapper):
    def __init__(
        self,
        model_name: str,
        sep: str = " ",
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.model_prompts = (
            validate_task_to_prompt_name(model_prompts) if model_prompts else None
        )

    def _embed(
        self,
        texts: list[str],
        google_task_type: str | None = None,
        titles: list[str] | None = None,
        dimensionality: int | None = 768,
    ) -> list[list[float]]:
        """Embeds texts with a pre-trained, foundational model.
        From https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#generative-ai-get-text-embedding-python_vertex_ai_sdk
        """
        try:
            from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
        except ImportError:
            raise ImportError(
                "The `vertexai` package is required to run the google API, please install it using `pip install vertexai`"
            )
        model = TextEmbeddingModel.from_pretrained(self.model_name)
        if titles:
            # Allow title-only embeddings by replacing text with a space
            # Else Google throws google.api_core.exceptions.InvalidArgument: 400 The text content is empty.
            inputs = [
                TextEmbeddingInput(
                    text if text else " ", task_type=google_task_type, title=title
                )
                for text, title in zip(texts, titles)
            ]
        else:
            inputs = [
                TextEmbeddingInput(text, task_type=google_task_type) for text in texts
            ]
        kwargs = {"output_dimensionality": dimensionality} if dimensionality else {}
        try:
            embeddings = model.get_embeddings(inputs, **kwargs)
        # Except the very rare google.api_core.exceptions.InternalServerError
        except Exception as e:
            print("Retrying once after error:", e)
            embeddings = model.get_embeddings(inputs, **kwargs)
        return np.asarray([embedding.values for embedding in embeddings])

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        google_task_type = get_prompt_name(self.model_prompts, task_name, prompt_type)
        return self._embed(sentences, google_task_type=google_task_type)


name = "text-embedding-004"
google_emb_004 = ModelMeta(
    loader=partial(
        GoogleTextEmbeddingModel,
        model_name=name,
        model_prompts={
            "Classification": "CLASSIFICATION",
            "MultilabelClassification": "CLASSIFICATION",
            "Clustering": "CLUSTERING",
            "STS": "SIMILARITY",
            PromptType.query.value: "RETRIEVAL_QUERY",
            PromptType.passage.value: "RETRIEVAL_DOCUMENT",
        },
    ),
    name=name,
    languages=["eng-Latn"],
    open_weights=False,
    revision="1",  # revision is intended for implementation
    release_date=None,  # couldnt figure this out
    n_parameters=None,
    memory_usage=None,
    max_tokens=2048,
    embed_dim=768,
    license=None,
    similarity_fn_name="cosine",  # assumed
    framework=["API"],
    use_instuctions=True,
)
