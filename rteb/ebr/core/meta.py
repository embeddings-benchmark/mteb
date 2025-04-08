from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict

from ebr.core.base import RetrievalDataset, EmbeddingModel

# Tier 0: fully open (documents, queries, relevance)
# Tier 1: documents and queries released
# Tier 2: documents released
# Tier 3: fully held out
DATASET_TIER = Literal[0, 1, 2, 3]

EMBEDDING_DTYPES = Literal["float32", "int8", "binary"]
SIMILARITY_METRICS = Literal["cosine", "dot"]


def dataset_id(
    dataset_name: str
) -> str:
    return f"{dataset_name}"


def model_id(
    model_name: str,
    embd_dtype: str,
    embd_dim: int,
) -> str:
    return f"{model_name.replace('/', '__')}_{embd_dtype}_{embd_dim}d"


class DatasetMeta(BaseModel):
    """Dataset metadata object.

    Attributes:
        TODO
    """

    model_config: ConfigDict = ConfigDict(protected_namespaces=())

    loader: Callable[..., RetrievalDataset]
    dataset_name: str
    tier: DATASET_TIER = 3
    groups: dict[str, int] = {}
    reference: str | None = None

    def model_dump(self, **kwargs) -> dict[str, Any]:
        exclude = kwargs.pop("exclude", set()) | {"loader"}
        return super().model_dump(exclude=exclude, **kwargs)

    def model_dump_json(self, **kwargs) -> dict[str, Any]:
        exclude = kwargs.pop("exclude", set()) | {"loader"}
        return super().model_dump_json(exclude=exclude, **kwargs)

    def load_dataset(self, data_path: str, **kwargs):
        return self.loader(data_path, self, **kwargs)        

    @property
    def _id(self) -> str:
        return dataset_id(self.dataset_name)


class ModelMeta(BaseModel):
    """Model metadata object. Adapted from embeddings-benchmark/mteb/model_meta.py.

    Attributes:
        loader: the function that loads the model.
        name: The name of the model.
        embd_dtype: The data type of the embeddings produced by the model, e.g. `float32`.
        embd_dim: The dimension of the embeddings produced by the model, e.g. `1024`.
        num_params: The number of parameters in the model, e.g. `7_000_000` for a 7M parameter model.
        max_tokens: The maximum number of tokens the model can handle. 
        similarity: Similarity function, e.g. cosine, dot-product, etc.
        query_instruct: Prompt to prepend to the input for queries.
        corpus_instruct: Prompt to prepend to the input for documents.
    """

    model_config: ConfigDict = ConfigDict(protected_namespaces=())

    loader: Callable[..., EmbeddingModel]
    model_name: str
    embd_dtype: EMBEDDING_DTYPES | None = None
    embd_dim: int | None = None
    num_params: int | None = None
    max_tokens: int | None = None
    similarity: SIMILARITY_METRICS | None = None
    query_instruct: str | None = None
    corpus_instruct: str | None = None
    reference: str | None = None
    alias: str | None = None

    def model_dump(self, **kwargs) -> dict[str, Any]:
        exclude = kwargs.pop("exclude", set()) | {"loader"}
        return super().model_dump(exclude=exclude, **kwargs)

    def model_dump_json(self, **kwargs) -> dict[str, Any]:
        exclude = kwargs.pop("exclude", set()) | {"loader"}
        return super().model_dump_json(exclude=exclude, **kwargs)

    def load_model(self, **kwargs) -> EmbeddingModel:
        return self.loader(self, **kwargs)

    @property
    def _id(self) -> str:
        return model_id(
            self.model_name,
            self.embd_dtype,
            self.embd_dim
        )
