from ebr.core.base import EmbeddingModel
from ebr.utils.lazy_import import LazyImport
from ebr.core.meta import ModelMeta

BGEM3FlagModel = LazyImport("FlagEmbedding", attribute="BGEM3FlagModel")


class BGEM3EmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_meta: ModelMeta,
        **kwargs
    ):
        super().__init__(model_meta, **kwargs)
        self._model = BGEM3FlagModel(
            model_name_or_path=model_meta.model_name,
        )

    def embed(self, data: list[str], input_type: str) -> list[list[float]]:
        result = self._model.encode(sentences=data, batch_size=12)['dense_vecs']
        return [[float(str(x)) for x in result[i]] for i in range(len(result))]


bge_m3 = ModelMeta(
    loader=BGEM3EmbeddingModel,
    model_name='BAAI/bge-m3',
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=8192,
    similarity="cosine",
    reference="https://huggingface.co/BAAI/bge-m3"
)
#
# bge_m3_unsupervised = ModelMeta(
#     loader=BGEM3EmbeddingModel,
#     model_name='BAAI/bge-m3-unsupervised',
#     embd_dtype="float32",
#     embd_dim=1024,
#     max_tokens=8192,
#     similarity="cosine",
#     reference="https://huggingface.co/BAAI/bge-m3-unsupervised"
# )
#
# bge_m3_retromae = ModelMeta(
#     loader=BGEM3EmbeddingModel,
#     model_name='BAAI/bge-m3-retromae',
#     embd_dtype="float32",
#     max_tokens=8192,
#     similarity="cosine",
#     reference="https://huggingface.co/BAAI/bge-m3-retromae"
# )
#
# bge_large_en_v15 = ModelMeta(
#     loader=BGEM3EmbeddingModel,
#     model_name='BAAI/bge-large-en-v1.5',
#     embd_dtype="float32",
#     embd_dim=1024,
#     max_tokens=512,
#     similarity="cosine",
#     reference="https://huggingface.co/BAAI/bge-large-en-v1.5"
# )
#
# bge_base_en_v15 = ModelMeta(
#     loader=BGEM3EmbeddingModel,
#     model_name='BAAI/bge-base-en-v1.5',
#     embd_dtype="float32",
#     embd_dim=768,
#     max_tokens=512,
#     similarity="cosine",
#     reference="https://huggingface.co/BAAI/bge-base-en-v1.5"
# )
#
# bge_small_en_v15 = ModelMeta(
#     loader=BGEM3EmbeddingModel,
#     model_name='BAAI/bge-small-en-v1.5',
#     embd_dtype="float32",
#     embd_dim=384,
#     max_tokens=512,
#     similarity="cosine",
#     reference="https://huggingface.co/BAAI/bge-small-en-v1.5"
# )
