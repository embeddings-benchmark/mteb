"""Sionic AI embedding models."""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

comsat_embed_ja_8b_preview = ModelMeta(
    # Plain sentence-transformers load: the instruct query prompt
    # ("Instruct: {task}\nQuery:") ships in the model's
    # config_sentence_transformers.json and is picked up automatically.
    loader=SentenceTransformerEncoderWrapper,
    name="sionic-ai/comsat-embed-ja-8b-preview",
    model_type=["dense"],
    revision="5dccac94bcba067e2a3cf413e8adc338e15a8d59",
    release_date="2026-07-03",
    languages=["jpn-Jpan", "eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    n_parameters=7_567_295_488,
    n_embedding_parameters=621_219_840,
    memory_usage_mb=14433,
    max_tokens=8192,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/sionic-ai/comsat-embed-ja-8b-preview",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    adapted_from="Qwen/Qwen3-Embedding-8B",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    # Trained on ~1.5M Japanese examples (model card); the source composition
    # is not documented, so possible mteb-task overlap is unknown.
    training_datasets=None,
)
