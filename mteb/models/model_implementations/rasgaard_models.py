import numpy as np

from mteb.models.model_implementations.model2vec_models import Model2VecModel
from mteb.models.model_meta import ModelMeta, ScoringFunction

potion_base_8m = ModelMeta(
    loader=Model2VecModel,  # type: ignore
    name="rasgaard/m2v-dfm-large",
    languages=["dan-Latn"],
    open_weights=True,
    revision="387897cfb09992e6d45ea9cd7b28b9fcf119e23a",
    release_date="2025-10-08",
    n_parameters=22893312,
    memory_usage_mb=87,
    max_tokens=np.inf,
    embed_dim=256,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers"],
    reference="https://huggingface.co/rasgaard/m2v-dfm-large",
    use_instructions=False,
    adapted_from="KennethEnevoldsen/dfm-sentence-encoder-large",
    superseded_by=None,
    training_datasets=set(),  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",  # distilled on this
    citation="""@article{minishlab2024model2vec,
    author = {Tulkens, Stephan and {van Dongen}, Thomas},
    title = {Model2Vec: Fast State-of-the-Art Static Embeddings},
    year = {2024},
    url = {https://github.com/MinishLab/model2vec}
}""",
)
