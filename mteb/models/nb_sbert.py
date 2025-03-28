from __future__ import annotations

from mteb.model_meta import ModelMeta

nb_sbert = ModelMeta(
    name="NbAiLab/nb-sbert-base",
    languages=["nno-Latn", "nob-Latn", "swe-Latn", "dan-Latn"],
    open_weights=True,
    revision="b95656350a076aeafd2d23763660f80655408cc6",
    release_date="2022-11-23",
    n_parameters="1_780_000_000",
    memory_usage_mb=197,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=75,
    reference="https://huggingface.co/NbAiLab/nb-sbert-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/NbAiLab/mnli-norwegian",
    training_datasets={},
)
