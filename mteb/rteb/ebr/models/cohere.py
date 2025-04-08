from __future__ import annotations
from typing import Any, TYPE_CHECKING

from ebr.core.meta import ModelMeta
from ebr.core.base import APIEmbeddingModel
from ebr.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import cohere
else:
    cohere = LazyImport("cohere")


class CohereEmbeddingModel(APIEmbeddingModel):

    def __init__(
        self,
        model_meta: ModelMeta,
        api_key: str | None = None,
        num_retries: int | None = None,
        **kwargs
    ):
        super().__init__(
            model_meta,
            api_key=api_key,
            num_retries=num_retries,
            **kwargs
        )
        self._client = None

    @property
    def client(self) -> cohere.ClientV2:
        if not self._client:
            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    @property
    def embedding_type(self) -> str:
        if self.embd_dtype == "float32":
            return "float"
        else:
            raise NotImplementedError

    def embed(self, data: str, input_type: str) -> list[list[float]]:
        
        return getattr(self.client.embed(
            model=self.model_name,
            texts=data,
            input_type="search_query" if input_type == "query" else "search_document",
            embedding_types=[self.embedding_type]
        ).embeddings, self.embedding_type)

    @staticmethod
    def rate_limit_error_type() -> type:
        return cohere.errors.too_many_requests_error.TooManyRequestsError


"""
embed_multilingual_v3_0 = ModelMeta(
    loader=CohereEmbeddingModel,
    model_name="embed-multilingual-v3.0",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=512,
    similarity="cosine",
    reference="https://docs.cohere.com/v2/docs/cohere-embed"
)
"""

