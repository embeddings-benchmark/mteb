from ebr.core.base import EmbeddingModel
from ebr.core.meta import ModelMeta, model_id
from ebr.models.cohere import *
from ebr.models.openai import *
from ebr.models.sentence_transformers import *
from ebr.models.voyageai import *
from ebr.models.bgem3 import *
from ebr.models.gritlm import *
from ebr.models.google import *


MODEL_REGISTRY: dict[str, ModelMeta] = {}
for name in dir():
    meta = eval(name)
    # Explicitly exclude `LazyImport` instances since the latter check invokes the import.
    if not isinstance(meta, LazyImport) and isinstance(meta, ModelMeta):
        MODEL_REGISTRY[meta._id] = eval(name)


def get_embedding_model(
    model_name: str, 
    embd_dim: int,
    embd_dtype: str,
    **kwargs
) -> EmbeddingModel:
    key = model_id(model_name, embd_dim, embd_dtype)
    #TODO: add logic to dynamically load missing model
    return MODEL_REGISTRY[key].load_model(**kwargs)
