from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

code_embed = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="divakantpandey/code-embed",
    model_type=["dense"],
    languages=["python-Code", "eng-Latn"],
    revision="0f0286c84171deb4d9f1fbfb126ad207d5ec42a6",
    release_date="2026-10-18",
    modalities=["text"],
    n_parameters=78_974_976,
    n_embedding_parameters=78_974_976,
    memory_usage_mb=301,
    max_tokens=8192,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/divakantpandey/code-embed",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/divakantpandey/code-embed",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    citation=None,
)
