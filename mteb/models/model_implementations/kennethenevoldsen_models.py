from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import (
    sentence_transformers_loader,
)

dfm_enc_large = ModelMeta(
    loader=sentence_transformers_loader,
    name="KennethEnevoldsen/dfm-sentence-encoder-large",
    model_type=["dense"],
    languages=["dan-Latn"],
    open_weights=True,
    revision="132c53391e7a780dc6a2f9a03724d0158fe7122c",
    release_date="2023-07-12",
    n_parameters=355087360,
    memory_usage_mb=1554,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/KennethEnevoldsen/dfm-sentence-encoder-large",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from="chcaa/dfm-encoder-large-v1",
    training_datasets=set(),  # just contrastive pre-training
    public_training_code="https://huggingface.co/KennethEnevoldsen/dfm-sentence-encoder-large#hyperparameters",
    citation="""@article{enevoldsenScandinavianEmbeddingBenchmarks2024,
    title = {The {Scandinavian} {Embedding} {Benchmarks}: {Comprehensive} {Assessment} of {Multilingual} and {Monolingual} {Text} {Embedding}},
    shorttitle = {The {Scandinavian} {Embedding} {Benchmarks}},
    url = {https://openreview.net/forum?id=pJl_i7HIA72},
    language = {en},
    urldate = {2024-04-12},
    author = {Enevoldsen, Kenneth and Kardos, Márton and Muennighoff, Niklas and Nielbo, Kristoffer},
    month = feb,
    year = {2024},
}
""",
    public_training_data="https://huggingface.co/datasets/danish-foundation-models/danish-gigaword",  # paragraphs extracted from Danish Gigaword
)

dfm_enc_med = ModelMeta(
    loader=sentence_transformers_loader,
    name="KennethEnevoldsen/dfm-sentence-encoder-medium",
    model_type=["dense"],
    languages=["dan-Latn"],
    open_weights=True,
    revision="701bce95d499fa97610d57e8823c54fd1fb79930",
    release_date="2023-07-12",
    n_parameters=124445952,
    memory_usage_mb=475,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/KennethEnevoldsen/dfm-sentence-encoder-medium",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    training_datasets=set(),  # just contrastive pre-training
    citation="""@article{enevoldsenScandinavianEmbeddingBenchmarks2024,
    title = {The {Scandinavian} {Embedding} {Benchmarks}: {Comprehensive} {Assessment} of {Multilingual} and {Monolingual} {Text} {Embedding}},
    shorttitle = {The {Scandinavian} {Embedding} {Benchmarks}},
    url = {https://openreview.net/forum?id=pJl_i7HIA72},
    language = {en},
    urldate = {2024-04-12},
    author = {Enevoldsen, Kenneth and Kardos, Márton and Muennighoff, Niklas and Nielbo, Kristoffer},
    month = feb,
    year = {2024},
}
""",
    public_training_data=None,
)
