from __future__ import annotations

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

intelli_embed_v3 = ModelMeta(
    loader=sentence_transformers_loader,
    name="serhiiseletskyi/intelli-embed-v3",
    model_type=["dense"],
    revision="d63e20fa8dcc133d21c5d63da921c2df73bca1ad",
    release_date="2025-02-21",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch", "ONNX", "safetensors"],
    n_parameters=567_754_752,
    n_embedding_parameters=256_002_048,
    memory_usage_mb=2166,
    max_tokens=8194,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/serhiiseletskyi/intelli-embed-v3",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    adapted_from="Snowflake/snowflake-arctic-embed-l-v2.0",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "NQ",
        "NQHardNegatives",
        "HotPotQA",
        "HotpotQAHardNegatives",
        "FEVER",
        "FEVERHardNegatives",
    },
)
