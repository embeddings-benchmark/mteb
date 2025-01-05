from __future__ import annotations

import json
import logging
from functools import partial
from typing import Any

import numpy as np

from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_package

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


class AmazonWrapper(Wrapper):
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
                body=json.dumps({"inputText": sentence}),
                modelId=self._model_id,
                accept="application/json",
                contentType="application/json",
            )
            all_embeddings.append(self._to_numpy(response))

        return np.array(all_embeddings)

    def _to_numpy(self, embedding_response) -> np.ndarray:
        response = json.loads(embedding_response.get("body").read())
        return np.array(response["embedding"])


amazon_titan_embed_text_v1 = ModelMeta(
    name="amazon/titan-embed-text-v1",
    revision="1",
    release_date="2023-09-27",
    languages=None,  # supported languages not specified
    loader=partial(AmazonWrapper, model_id="amazon.titan-embed-text-v1"),
    max_tokens=8192,
    embed_dim=1536,
    open_weights=False,
    n_parameters=None,
    memory_usage=None,
    license=None,
    reference="https://aws.amazon.com/about-aws/whats-new/2023/09/amazon-titan-embeddings-generally-available/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
)

amazon_titan_embed_text_v2 = ModelMeta(
    name="amazon/titan-embed-text-v2",
    revision="1",
    release_date="2024-04-30",
    languages=None,  # supported languages not specified
    loader=partial(AmazonWrapper, model_id="amazon.titan-embed-text-v2:0"),
    max_tokens=8192,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    memory_usage=None,
    license=None,
    reference="https://aws.amazon.com/about-aws/whats-new/2024/04/amazon-titan-text-embeddings-v2-amazon-bedrock/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=False,
)
