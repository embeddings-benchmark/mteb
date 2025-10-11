from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

greennode_embedding_large_vn_v1_training_data = {
    "GreenNodeTableMarkdownRetrieval",
}

greennode_embedding_large_vn_v1 = ModelMeta(
    name="GreenNode/GreenNode-Embedding-Large-VN-V1",
    revision="660def1f6e1c8ecdf39f6f9c95829e3cf0cef837",
    release_date="2024-04-11",
    languages=[
        "vie-Latn",
    ],
    loader=sentence_transformers_loader,
    open_weights=True,
    n_parameters=568_000_000,
    memory_usage_mb=2167,
    embed_dim=1024,
    license="cc-by-4.0",
    max_tokens=8194,
    reference="https://huggingface.co/GreenNode/GreenNode-Embedding-Large-VN-V1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/GreenNode/GreenNode-Table-Markdown-Retrieval-VN",
    training_datasets=greennode_embedding_large_vn_v1_training_data,
    adapted_from="BAAI/bge-m3",
)

greennode_embedding_large_vn_mixed_v1 = ModelMeta(
    name="GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1",
    revision="1d3dddb3862292dab4bd3eddf0664c0335ad5843",
    release_date="2024-04-11",
    languages=[
        "vie-Latn",
    ],
    loader=sentence_transformers_loader,
    open_weights=True,
    n_parameters=568_000_000,
    memory_usage_mb=2167,
    embed_dim=1024,
    license="cc-by-4.0",
    max_tokens=8194,
    reference="https://huggingface.co/GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/GreenNode/GreenNode-Table-Markdown-Retrieval-VN",
    training_datasets=greennode_embedding_large_vn_v1_training_data,
    adapted_from="BAAI/bge-m3",
)

aiteamvn_vietnamese_embeddings = ModelMeta(
    name="AITeamVN/Vietnamese_Embedding",
    revision="fcbbb905e6c3757d421aaa5db6fd7c53d038f6fb",
    release_date="2024-03-17",
    languages=[
        "vie-Latn",
    ],
    loader=sentence_transformers_loader,
    open_weights=True,
    n_parameters=568_000_000,
    memory_usage_mb=2166,
    embed_dim=1024,
    license="cc-by-4.0",
    max_tokens=8194,
    reference="https://huggingface.co/AITeamVN/Vietnamese_Embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from="BAAI/bge-m3",
)

hiieu_halong_embedding = ModelMeta(
    name="hiieu/halong_embedding",
    revision="b57776031035f70ed2030d2e35ecc533eb0f8f71",
    release_date="2024-07-06",
    languages=[
        "vie-Latn",
    ],
    loader=sentence_transformers_loader,
    use_instructions=False,
    open_weights=True,
    n_parameters=278_000_000,
    memory_usage_mb=1061,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=514,
    reference="https://huggingface.co/hiieu/halong_embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from="intfloat/multilingual-e5-base",
)

sup_simcse_vietnamese_phobert_base_ = ModelMeta(
    name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
    revision="608779b86741a8acd8c8d38132974ff04086b138",
    release_date="2021-05-26",
    languages=[
        "vie-Latn",
    ],
    loader=sentence_transformers_loader,
    use_instructions=False,
    open_weights=True,
    n_parameters=135_000_000,
    memory_usage_mb=517,
    max_tokens=256,
    embed_dim=768,
    license="apache-2.0",
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
    similarity_fn_name="cosine",
    training_datasets=None,
)

bkai_foundation_models_vietnamese_bi_encoder = ModelMeta(
    name="bkai-foundation-models/vietnamese-bi-encoder",
    revision="84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0",
    release_date="2023-09-09",
    languages=[
        "vie-Latn",
    ],
    loader=sentence_transformers_loader,
    use_instructions=False,
    open_weights=True,
    n_parameters=135_000_000,
    memory_usage_mb=515,
    max_tokens=256,
    embed_dim=768,
    license="apache-2.0",
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder",
    similarity_fn_name="cosine",
    training_datasets=None,
)
