from mteb.model_meta import ModelMeta, sentence_transformers_loader

kowshik24_bangla_embedding_model = ModelMeta(
    loader=sentence_transformers_loader,
    name="Kowshik24/bangla-sentence-transformer-ft-matryoshka-paraphrase-multilingual-mpnet-base-v2",
    languages=["ben-Beng"],  # Bengali using Bengali script
    open_weights=True,
    revision="6689c21e69be5950596bad084457cbaa138728d8",
    release_date="2025-11-10",  
    n_parameters=278_000_000,   
    memory_usage_mb=1061,       
    embed_dim=768,
    license="apache-2.0",
    max_tokens=128,
    reference="https://huggingface.co/Kowshik24/bangla-sentence-transformer-ft-matryoshka-paraphrase-multilingual-mpnet-base-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/kowshik24/Bangla-Embedding",
    public_training_data="https://huggingface.co/datasets/sartajekram/BanglaRQA",
    training_datasets=set(),
)
