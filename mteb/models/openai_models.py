from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np

from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_package

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


class OpenAIWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        tokenizer_name: str = "cl100k_base",  # since all models use this tokenizer now
        embed_dim: int | None = None,
        **kwargs,
    ) -> None:
        requires_package(self, "openai", "Openai text embedding")
        from openai import OpenAI

        self._client = OpenAI()
        self._model_name = model_name
        self._embed_dim = embed_dim
        self._max_seq_length = max_seq_length
        self._tokenizer_name = tokenizer_name

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        requires_package(self, "openai", "Openai text embedding")
        import tiktoken
        from openai import NotGiven

        if self._model_name == "text-embedding-ada-002" and self._embed_dim is not None:
            logger.warning(
                "Reducing embedding size available only for text-embedding-3-* models"
            )

        trimmed_sentences = []
        for sentence in sentences:
            encoding = tiktoken.get_encoding(self._tokenizer_name)
            encoded_sentence = encoding.encode(sentence)
            if len(encoded_sentence) > self._max_seq_length:
                trimmed_sentence = encoding.decode(
                    encoded_sentence[: self._max_seq_length]
                )
                trimmed_sentences.append(trimmed_sentence)
            else:
                trimmed_sentences.append(sentence)

        max_batch_size = 2048
        sublists = [
            trimmed_sentences[i : i + max_batch_size]
            for i in range(0, len(trimmed_sentences), max_batch_size)
        ]

        all_embeddings = []

        for sublist in sublists:
            response = self._client.embeddings.create(
                input=sublist,
                model=self._model_name,
                encoding_format="float",
                dimensions=self._embed_dim or NotGiven(),
            )
            all_embeddings.extend(self._to_numpy(response))

        return np.array(all_embeddings)

    def _to_numpy(self, embedding_response) -> np.ndarray:
        return np.array([e.embedding for e in embedding_response.data])


text_embedding_3_small = ModelMeta(
    name="text-embedding-3-small",
    revision="1",
    release_date="2024-01-25",
    languages=None,  # supported languages not specified
    loader=partial(OpenAIWrapper, model_name="text-embedding-3-small"),
    max_tokens=8191,
    embed_dim=1536,
    open_weights=False,
    n_parameters=None,
    memory_usage=None,
    license=None,
    reference="https://openai.com/index/new-embedding-models-and-api-updates/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
)
text_embedding_3_large = ModelMeta(
    name="text-embedding-3-large",
    revision="1",
    release_date="2024-01-25",
    languages=None,  # supported languages not specified
    loader=partial(OpenAIWrapper, model_name="text-embedding-3-large"),
    max_tokens=8191,
    embed_dim=3072,
    open_weights=False,
    framework=["API"],
    use_instructions=False,
    n_parameters=None,
    memory_usage=None,
)
text_embedding_ada_002 = ModelMeta(
    name="text-embedding-ada-002",
    revision="1",
    release_date="2022-12-15",
    languages=None,  # supported languages not specified
    loader=partial(OpenAIWrapper, model_name="text-embedding-ada-002"),
    max_tokens=8191,
    embed_dim=1536,
    open_weights=False,
    framework=["API"],
    use_instructions=False,
    n_parameters=None,
    memory_usage=None,
)
