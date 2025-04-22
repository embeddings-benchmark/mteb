from __future__ import annotations
from functools import partial
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
from mteb.model_meta import ModelMeta, sentence_transformers_loader

codemodernbert_crow_meta = ModelMeta(
    name="Shuu12121/CodeSearch-ModernBERT-Crow-Plus",
    revision="fd3d662ca9ba8d0e6be981cb3be21aee7a80096e",
    languages=["eng-Latn"],
    loader=partial(
        sentence_transformers_loader,
        model="Shuu12121/CodeSearch-ModernBERT-Crow-Plus",
        revision="fd3d662ca9ba8d0e6be981cb3be21aee7a80096e",
    ),
    open_weights=True,
    release_date="2025-04-21",
    n_parameters=151668480,
    memory_usage_mb=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=1028,
    reference="https://huggingface.co/Shuu12121/CodeSearch-ModernBERT-Crow-Plus",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "code-search-net/code_search_net": ["train"],
        "Shuu12121/python-codesearch-filtered": ["train"],
        "Shuu12121/java-codesearch-filtered": ["train"],
        "Shuu12121/javascript-codesearch-filtered": ["train"],
        "Shuu12121/ruby-codesearch-filtered": ["train"],
        "Shuu12121/rust-codesearch-filtered": ["train"],
    },
)
