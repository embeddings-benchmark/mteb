from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

codemodernbert_crow_meta = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="Shuu12121/CodeSearch-ModernBERT-Crow-Plus",
        revision="044a7a4b552f86e284817234c336bccf16f895ce",
    ),
    name="Shuu12121/CodeSearch-ModernBERT-Crow-Plus",
    languages=["eng-Latn"],
    open_weights=True,
    revision="044a7a4b552f86e284817234c336bccf16f895ce",
    release_date="2025-04-21",
    n_parameters=151668480,
    memory_usage_mb=607,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=1024,
    reference="https://huggingface.co/Shuu12121/CodeSearch-ModernBERT-Crow-Plus",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "CodeSearchNetRetrieval": [],
        # "code-search-net/code_search_net": ["train"],
        # "Shuu12121/python-codesearch-filtered": ["train"],
        # "Shuu12121/java-codesearch-filtered": ["train"],
        # "Shuu12121/javascript-codesearch-filtered": ["train"],
        # "Shuu12121/ruby-codesearch-filtered": ["train"],
        # "Shuu12121/rust-codesearch-filtered": ["train"],
    },
)
