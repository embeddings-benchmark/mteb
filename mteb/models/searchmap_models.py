from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

# Define task instructions with specific task names
task_instructions = {
    "Classification": "Generate a representation for this text that can be used for classification:",
    "Clustering": "Generate a representation for this text that can be used for clustering:",
    "Retrieval": "Generate a representation for this text that can be used for retrieval:",
    "STS": "Generate a representation for this text that can be used for semantic similarity:",
    "PairClassification": "Generate a representation for this text pair that can be used for classification:",
    "Reranking": "Generate a representation for this text that can be used for reranking:",
    "Summarization": "Generate a representation for this text that can be used for summarization:",
}

searchmap_preview = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="VPLabs/SearchMap_Preview",
        revision="69de17ef48278ed08ba1a4e65ead8179912b696e",
        model_prompts=task_instructions,
    ),
    name="VPLabs/SearchMap_Preview",
    revision="69de17ef48278ed08ba1a4e65ead8179912b696e",
    languages=["eng_Latn"],
    open_weights=True,
    use_instructions=True,
    release_date="2025-03-05",
    n_parameters=435_000_000,
    memory_usage_mb=1660,
    embed_dim=4096,
    license="mit",
    max_tokens=8192,
    reference="https://huggingface.co/VPLabs/SearchMap_Preview",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from="NovaSearch/stella_en_400M_v5",
)
