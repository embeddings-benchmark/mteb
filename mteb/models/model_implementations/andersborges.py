import numpy as np

from mteb.models.model_implementations.model2vec_models import Model2VecModel
from mteb.models.model_meta import ModelMeta, ScoringFunction

model2vecdk = ModelMeta(
    loader=Model2VecModel,
    name="andersborges/model2vecdk",
    model_type=["dense"],
    languages=["dan-Latn"],
    open_weights=True,
    revision="cb576c78dcc1b729e4612645f61db59929d69e61",
    release_date="2025-11-21",
    n_parameters=48042496,
    memory_usage_mb=183,
    max_tokens=np.inf,
    embed_dim=256,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers"],
    reference="https://huggingface.co/andersborges/model2vecdk",
    use_instructions=False,
    adapted_from="https://huggingface.co/jealk/TTC-L2V-supervised-2",
    superseded_by=None,
    training_datasets=set(),  # distilled
    public_training_code="https://github.com/andersborges/dkmodel2vec",
    public_training_data="https://huggingface.co/datasets/DDSC/nordic-embedding-training-data",
    citation="""@article{minishlab2024model2vec,
  author = {Tulkens, Stephan and {van Dongen}, Thomas},
  title = {Model2Vec: Fast State-of-the-Art Static Embeddings},
  year = {2024},
  url = {https://github.com/MinishLab/model2vec}
}""",
)


model2vecdk_stem = ModelMeta(
    loader=Model2VecModel,
    name="andersborges/model2vecdk-stem",
    model_type=["dense"],
    languages=["dan-Latn"],
    open_weights=True,
    revision="cb576c78dcc1b729e4612645f61db59929d69e61",
    release_date="2025-11-21",
    n_parameters=48578560,
    memory_usage_mb=185,
    max_tokens=np.inf,
    embed_dim=256,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers"],
    reference="https://huggingface.co/andersborges/model2vecdk",
    use_instructions=False,
    adapted_from="https://huggingface.co/jealk/TTC-L2V-supervised-2",
    superseded_by=None,
    training_datasets=set(),  # distilled
    public_training_code="https://github.com/andersborges/dkmodel2vec",
    public_training_data="https://huggingface.co/datasets/DDSC/nordic-embedding-training-data",
    citation="""@article{minishlab2024model2vec,
  author = {Tulkens, Stephan and {van Dongen}, Thomas},
  title = {Model2Vec: Fast State-of-the-Art Static Embeddings},
  year = {2024},
  url = {https://github.com/MinishLab/model2vec}
}""",
)
