from __future__ import annotations
from typing import Any, TYPE_CHECKING

from ebr.core.meta import ModelMeta
from ebr.core.base import APIEmbeddingModel
from ebr.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import openai
    import tiktoken
else:
    openai = LazyImport("openai")
    tiktoken = LazyImport("tiktoken")


class OpenAIEmbeddingModel(APIEmbeddingModel):

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
        self._tokenizer = None

    @property
    def client(self) -> openai.OpenAI:
        if not self._client:
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def embed(self, data: str, input_type: str) -> list[list[float]]:
        tokens = [self.tokenizer.encode(text, disallowed_special=()) for text in data]
        if self.max_tokens:
            for n, tok in enumerate(tokens):
                if len(tok) > self.max_tokens:
                    tokens[n] = tok[:self.max_tokens]
        result = self.client.embeddings.create(
            input=tokens,
            model=self.model_name,
            dimensions=self.embd_dim
        )
        embeddings = [d.embedding for d in result.data]
        return embeddings

    @staticmethod
    def rate_limit_error_type() -> type:
        return openai.RateLimitError

    @staticmethod
    def service_error_type() -> type:
        return openai.InternalServerError



text_embedding_3_large = ModelMeta(
    loader=OpenAIEmbeddingModel,
    model_name="text-embedding-3-large",
    embd_dtype="float32",
    embd_dim=3072,
    max_tokens=8191,
    similarity="cosine",
    reference="https://platform.openai.com/docs/guides/embeddings"
)


text_embedding_3_large_512d = ModelMeta(
    loader=OpenAIEmbeddingModel,
    model_name="text-embedding-3-large",
    embd_dtype="float32",
    embd_dim=512,
    max_tokens=8191,
    similarity="cosine",
    reference="https://platform.openai.com/docs/guides/embeddings"
)


text_embedding_3_small = ModelMeta(
    loader=OpenAIEmbeddingModel,
    model_name="text-embedding-3-small",
    embd_dtype="float32",
    embd_dim=1536,
    max_tokens=8191,
    similarity="cosine",
    reference="https://platform.openai.com/docs/guides/embeddings"
)


text_embedding_3_small_512d = ModelMeta(
    loader=OpenAIEmbeddingModel,
    model_name="text-embedding-3-small",
    embd_dtype="float32",
    embd_dim=512,
    max_tokens=8191,
    similarity="cosine",
    reference="https://platform.openai.com/docs/guides/embeddings"
)
