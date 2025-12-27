from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
)

nb_sbert = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,  # type: ignore[arg-type]
    name="NbAiLab/nb-sbert-base",
    model_type=["dense"],
    languages=["nno-Latn", "nob-Latn", "swe-Latn", "dan-Latn"],
    open_weights=True,
    revision="b95656350a076aeafd2d23763660f80655408cc6",
    release_date="2022-11-23",
    n_parameters=1_780_000_000,
    memory_usage_mb=678,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=75,
    reference="https://huggingface.co/NbAiLab/nb-sbert-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/NbAiLab/mnli-norwegian",
    training_datasets=set(),
)

nb_bert_large = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,  # type: ignore[arg-type]
    name="NbAiLab/nb-bert-large",
    model_type=["dense"],
    languages=["nno-Latn", "nob-Latn"],
    open_weights=True,
    revision="f9d0fc184adab4dc354d85e1854b7634540d7550",
    release_date="2021-04-29",
    n_parameters=355087360,
    memory_usage_mb=1359,
    embed_dim=1024,
    license="cc-by-4.0",
    max_tokens=512,
    reference="https://huggingface.co/NbAiLab/nb-bert-large",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data="https://huggingface.co/NbAiLab/nb-bert-large#training-data",
    training_datasets=set(),
)

nb_bert_base = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,  # type: ignore[arg-type]
    name="NbAiLab/nb-bert-base",
    model_type=["dense"],
    languages=["nno-Latn", "nob-Latn"],
    open_weights=True,
    revision="9417c3f62a3adc99f17ff92bff446f35d011f994",
    release_date="2021-01-13",
    n_parameters=177853440,
    memory_usage_mb=681,
    embed_dim=768,
    license="cc-by-4.0",
    max_tokens=512,
    reference="https://huggingface.co/NbAiLab/nb-bert-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data="https://huggingface.co/NbAiLab/nb-bert-base#training-data",
    training_datasets=set(),
)
