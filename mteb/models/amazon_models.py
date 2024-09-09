from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np
import json

from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class AmazonWrapper:
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

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        requires_package(self, "boto3", "Amazon Bedrock")

        all_embeddings = []

        for sentence in sentences:
            response = self._client.invoke_model(
                body=json.dumps({
                    "inputText": sentence
                }),
                modelId=self._model_id,
                accept="application/json",
                contentType="application/json"
            )
            all_embeddings.append(self._to_numpy(response))

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
        return np.array(response['embedding'])


amazon_titan_embed_text_v1 = ModelMeta(
    name="amazon-titan-embed-text-v1",
    revision="1",
    release_date=None,
    languages=None,  # supported languages not specified
    loader=partial(AmazonWrapper, model_id="amazon.titan-embed-text-v1"),
    max_tokens=8192,
    embed_dim=None,
    open_source=False,
)

amazon_titan_embed_text_v2 = ModelMeta(
    name="amazon-titan-embed-text-v2",
    revision="1",
    release_date=None,
    languages=None,  # supported languages not specified
    loader=partial(AmazonWrapper, model_id="amazon.titan-embed-text-v2:0"),
    max_tokens=8192,
    embed_dim=None,
    open_source=False,
)
