from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np
import json

import mteb
from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class CohereBedrockWrapper:
    def __init__(self, model_id: str, **kwargs) -> None:
        requires_package(self, "boto3", "Amazon Bedrock")
        import boto3
        boto3_session = boto3.session.Session()
        region_name = boto3_session.region_name
        self._client = boto3.client(
            "bedrock-runtime",
            region_name,
        )
        self._model_id = model_id

    def encode(self, sentences: list[str],
               prompt_name: str | None = None,
               cohere_task_type: str = "search_document",
               **kwargs: Any) -> np.ndarray:
        requires_package(self, "boto3", "Amazon Bedrock")

        if prompt_name:
            task = mteb.get_task(prompt_name)
            task_type = task.metadata.type
            if task_type in ["Classification", "MultilabelClassification"]:
                cohere_task_type = "classification"
            elif task_type == "Clustering":
                cohere_task_type = "clustering"

        max_batch_size = 96
        sublists = [
            sentences[i: i + max_batch_size]
            for i in range(0, len(sentences), max_batch_size)
        ]

        all_embeddings = []

        for sublist in sublists:
            response = self._client.invoke_model(
                body=json.dumps({
                    "texts": [sent[:2048] for sent in sublist],
                    "input_type": cohere_task_type}),
                modelId=self._model_id,
                accept="*/*",
                contentType="application/json"
            )
            all_embeddings.extend(self._to_numpy(response))

        return np.array(all_embeddings)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list[str]], **kwargs: Any
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus)
        return self.encode(sentences, **kwargs)

    def _to_numpy(self, embedding_response) -> np.ndarray:
        response = json.loads(embedding_response.get("body").read())
        return np.array(response['embeddings'])


cohere_embed_english_v3 = ModelMeta(
    name="cohere-embed-english-v3",
    revision="1",
    release_date=None,
    languages=None,  # supported languages not specified
    loader=partial(CohereBedrockWrapper,
                   model_id="cohere.embed-english-v3"),
    max_tokens=512,
    embed_dim=None,
    open_source=False,
)

cohere_embed_multilingual_v3 = ModelMeta(
    name="cohere-embed-multilingual-v3",
    revision="1",
    release_date=None,
    languages=None,  # supported languages not specified
    loader=partial(CohereBedrockWrapper,
                   model_id="cohere.embed-multilingual-v3"),
    max_tokens=512,
    embed_dim=None,
    open_source=False,
)
