from mteb.models import ModelMeta, sentence_transformers_loader

yuan_emb_zh_datasets = {
    "CMedQAv2-reranking",
    "DuRetrieval",
    "MMarcoReranking",
    "T2Reranking",
    "T2Retrieval",
}

# not in mteb
# "Multi-CPR":"http://github.com/Alibaba-NLP/Multi-CPR",

yuan_embedding_2_zh = ModelMeta(
    name="IEITYuan/Yuan-embedding-2.0-zh",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    languages=["zho-Hans"],
    open_weights=True,
    revision="b5ebcace6f4fc6e5a4d1852557eb2dc2d1040cee",
    release_date="2025-11-24",
    n_parameters=326000000,
    memory_usage_mb=1242,
    embed_dim=1792,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/IEITYuan/Yuan-embedding-2.0-zh",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=yuan_emb_zh_datasets,
)
