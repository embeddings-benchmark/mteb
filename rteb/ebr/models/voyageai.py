

from __future__ import annotations
from typing import Any, TYPE_CHECKING

from ebr.core.base import APIEmbeddingModel
from ebr.core.meta import ModelMeta
from ebr.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import voyageai
else:
    voyageai = LazyImport("voyageai")

class VoyageAIEmbeddingModel(APIEmbeddingModel):

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
    def client(self) -> voyageai.Client:
        if not self._client:
            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        result = self.client.embed(
            data,
            model=self.model_name,
            output_dimension=self.embd_dim,
            input_type=None
        )
        return result.embeddings

    @staticmethod
    def rate_limit_error_type() -> type:
        return voyageai.error.RateLimitError

    @staticmethod
    def service_error_type() -> type:
        return voyageai.error.ServiceUnavailableError



voyage_3 = ModelMeta(
    loader=VoyageAIEmbeddingModel,
    model_name="voyage-3",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=32_000,
    similarity="cosine",
    query_instruct="Represent the query for retrieving supporting documents: ",
    corpus_instruct="Represent the document for retrieval: ",
    reference="https://docs.voyageai.com/docs/embeddings"
)
