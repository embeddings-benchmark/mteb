from ebr.core.base import EmbeddingModel
from ebr.utils.lazy_import import LazyImport
from ebr.core.meta import ModelMeta

GritLM = LazyImport("gritlm", attribute="GritLM")


class GRITLMEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_meta: ModelMeta,
        **kwargs
    ):
        super().__init__(model_meta, **kwargs)
        self._model = GritLM(
            model_name_or_path= "GritLM/GritLM-7B",
            normalized= False,
            torch_dtype= model_meta.embd_dtype,
            mode= "embedding",
        )

    def embed(self, data: list[str], input_type: str) -> list[list[float]]:
        result = self._model.encode(sentences=data)
        return [[float(str(x)) for x in result[i]] for i in range(len(result))]


"""
gritlm_7b = ModelMeta(
    loader=GRITLMEmbeddingModel,
    model_name="GritLM/GritLM-7B",
    embd_dtype="float32",
    embd_dim=384,
    num_params=7_240_000,
    similarity="cosine",
    reference="https://huggingface.co/GritLM/GritLM-7B"
)

gritlm_8x7b = ModelMeta(
    loader=GRITLMEmbeddingModel,
    model_name="GritLM/GritLM-8x7B",
    embd_dtype="float32",
    embd_dim=384,
    num_params=46_700_000,
    similarity="cosine",
    reference="https://huggingface.co/GritLM/GritLM-8x7B"
)
"""

