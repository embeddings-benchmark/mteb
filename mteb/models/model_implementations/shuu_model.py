from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

codemodernbert_crow_meta = ModelMeta(
    loader=sentence_transformers_loader,
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
        "CodeSearchNetRetrieval",
        # "code-search-net/code_search_net",
        # "Shuu12121/python-codesearch-filtered",
        # "Shuu12121/java-codesearch-filtered",
        # "Shuu12121/javascript-codesearch-filtered",
        # "Shuu12121/ruby-codesearch-filtered",
        # "Shuu12121/rust-codesearch-filtered",
    },
)
