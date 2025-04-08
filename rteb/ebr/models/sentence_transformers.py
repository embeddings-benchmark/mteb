from ebr.core.base import EmbeddingModel
from ebr.utils.lazy_import import LazyImport
from ebr.core.meta import ModelMeta

SentenceTransformer = LazyImport("sentence_transformers", attribute="SentenceTransformer")


class SentenceTransformersEmbeddingModel(EmbeddingModel):

    def __init__(
        self,
        model_meta: ModelMeta,
        **kwargs
    ):
        super().__init__(model_meta, **kwargs)
        self._model = SentenceTransformer(f"{self.model_name_prefix}/{self.model_name}", trust_remote_code=True)

    def embed(self, data: str, input_type: str) -> list[list[float]]:
        return self._model.encode(data)

    @property
    def model_name_prefix(self) -> str:
        return "sentence-transformers"

    @property
    def _id(self) -> str:
        return f"{self.model_name_prefix}__{self._model_meta._id}"


all_MiniLM_L6_v2 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="all-MiniLM-L6-v2",
    embd_dtype="float32",
    embd_dim=384,
    num_params=22_700_000,
    max_tokens=256,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
)

"""
all_MiniLM_L12_v2 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    embd_dtype="float32",
    embd_dim=384,
    num_params=33_400_000,
    max_tokens=256,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2"
)


labse = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="sentence-transformers/LaBSE",
    embd_dtype="float32",
    embd_dim=768,
    num_params=471_000_000,
    max_tokens=512,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/LaBSE"
)


multi_qa_MiniLM_L6_cos_v1 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="sentence-transformer/multi-qa-MiniLM-L6-cos-v1",
    embd_dtype="float32",
    embd_dim=384,
    num_params=22_700_000,
    max_tokens=512,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)


all_mpnet_base_v2 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="sentence-transformers/all-mpnet-base-v2",
    embd_dtype="float32",
    embd_dim=768,
    num_params=109_000_000,
    max_tokens=384,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/all-mpnet-base-v2"
)


jina_embeddings_v2_base_en = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="jinaai/jina-embeddings-v2-base-en",
    embd_dtype="float32",
    embd_dim=768,
    num_params=137_000_000,
    max_tokens=8192,
    similarity="cosine",
    reference="https://huggingface.co/jinaai/jina-embeddings-v2-base-en"
)


jina_embeddings_v2_small_en = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="jinaai/jina-embeddings-v2-small-en",
    embd_dtype="float32",
    embd_dim=512,
    num_params=32_700_000,
    max_tokens=8192,
    similarity="cosine",
    reference="https://huggingface.co/jinaai/jina-embeddings-v2-small-en"
)
"""

