from mteb.model_meta import ModelMeta

my_model = ModelMeta(
    name="OrdalieTech/Solon-embeddings-mini-beta-1.1",
    languages=["fra-Latn"],                
    open_weights=True,
    revision="8e4ea66eb7eb6109b47b7d97d7556f154d9aec4a",
    release_date="2025-01-01",
    n_parameters=210_000_000,              
    embed_dim=768,                          
    license="apache-2.0",
    max_tokens=8192,                        
    reference="https://huggingface.co/OrdalieTech/Solon-embeddings-mini-beta-1.1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    training_datasets={
        "PleIAs/common_corpus": ["train"],
        "HuggingFaceFW/fineweb": ["train"],
        "OrdalieTech/wiki_fr": ["train"],
        "llm_synthetic_private": ["train"],
    },


)

