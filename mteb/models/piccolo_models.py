"""Piccolo Chinese embedding models by SenseNova"""

from __future__ import annotations

from mteb.model_meta import ModelMeta

piccolo_base_zh = ModelMeta(
    name="sensenova/piccolo-base-zh",
    languages=["zho-Hans"],
    open_weights=True,
    revision="47c0a63b8f667c3482e05b2fd45577bb19252196",
    release_date="2023-09-04",  # first commit
    n_parameters=None,
    memory_usage_mb=None,  # can't see on model card
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/sensenova/piccolo-base-zh",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,  # They don't specify
)

piccolo_large_zh_v2 = ModelMeta(
    name="sensenova/piccolo-large-zh-v2",
    languages=["zho-Hans"],
    open_weights=False,  # They "temporarily" removed it in may last year
    # "Due to certain internal company considerations"
    revision="05948c1d889355936bdf9db7d30df57dd78d25a3",
    release_date="2024-04-22",  # first commit
    n_parameters=None,
    memory_usage_mb=None,  # we don't know because they removed the model
    embed_dim=1024,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/sensenova/piccolo-large-zh-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,  # They don't say
)
