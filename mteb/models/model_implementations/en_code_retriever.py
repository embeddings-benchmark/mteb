from __future__ import annotations

from mteb.models import ModelMeta, sentence_transformers_loader
from mteb.types import PromptType

english_code_retriever = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts={
            PromptType.query.value: "search_query: ",
            PromptType.document.value: "search_document: ",
        },
    ),
    name="fyaronskiy/english_code_retriever",
    languages=["eng-Latn"],
    open_weights=True,
    revision="be653fab7d27a7348a0c2c3d16b9f92a7f10cb0c",
    release_date="2025-07-10",
    n_parameters=149_000_000,
    memory_usage_mb=568,
    embed_dim=768,
    license="mit",
    max_tokens=8192,
    reference="https://huggingface.co/fyaronskiy/english_code_retriever",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/code-search-net/code_search_net",
    training_datasets={"CodeSearchNet"},
)
