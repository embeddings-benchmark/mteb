"""Model implementation for VoiceCLAP models (LAION).

VoiceCLAP models are SentenceTransformer-based audio-text embedding models
that extend LCO-Embedding-Omni with LoRA adapters for voice/speech/emotion tasks.
"""

from __future__ import annotations

from mteb.models import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

voiceclap_large_v2 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={
        "trust_remote_code": True,
        "model_kwargs": {"torch_dtype": "bfloat16"},
    },
    name="laion/voiceclap-large-v2",
    languages=["eng-Latn"],
    revision="e3288bfe6d5772782cf74338737987a85cff0da9",
    release_date="2026-06-11",
    modalities=["audio", "text"],
    n_parameters=8_931_813_888,
    memory_usage_mb=18145,
    max_tokens=32768,
    embed_dim=3584,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/laion/voiceclap-large-v2",
    similarity_fn_name="cosine",
    use_instructions=False,
    adapted_from="LCO-Embedding/LCO-Embedding-Omni-7B",
    training_datasets=set(),
    citation=None,
)
