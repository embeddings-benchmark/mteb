from __future__ import annotations

from mteb.model_meta import ModelMeta

Qodo_Embed = ModelMeta(
    name="Qodo/Qodo-Embed-1-1.5B",
    languages=[
        "python-Code",
        "c++-Code",
        "c#-Code",
        "go-Code",
        "java-Code",
        "Javascript-Code",
        "php-Code",
        "ruby-Code",
        "typescript-Code",
    ],
    open_weights=True,
    revision="84bbef079b32e8823ec226d4e9e92902706b0eb6",
    release_date="2025-02-19",
    n_parameters=1_780_000_000,
    memory_usage_mb=6776,
    embed_dim=1536,
    license="QodoAI-Open-RAIL-M",
    max_tokens=32768,
    reference="https://huggingface.co/Qodo/Qodo-Embed-1-1.5B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
)
