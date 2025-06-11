from __future__ import annotations

from mteb.model_meta import ModelMeta
from mteb.models.bge_models import bge_full_data, bge_m3_training_data
from mteb.models.stella_models import stella_zh_datasets

ritrieve_zh_v1 = ModelMeta(
    name="richinfoai/ritrieve_zh_v1",
    languages=["zho-Hans"],
    open_weights=True,
    revision="f8d5a707656c55705027678e311f9202c8ced12c",
    release_date="2025-03-25",
    n_parameters=int(326 * 1e6),
    memory_usage_mb=1242,
    embed_dim=1792,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/richinfoai/ritrieve_zh_v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets={**stella_zh_datasets, **bge_full_data, **bge_m3_training_data},
)
