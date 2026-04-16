from mteb.models import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

saga_prompts = {
    # --- Default Task Types ---
    "STS": "task: semantic similarity | query: ",
    "Summarization": "task: semantic similarity | query: ",
    "Reranking": "task: semantic similarity | query: ",
    "BitextMining": "task: semantic similarity | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "Retrieval-query": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    
    # --- Exact Dataset Overrides ---
    "BornholmBitextMining": "",
    "NordicLangClassification": "task: clustering | query: ",
    
    # "faq", "quad", "hjerne" exceptions (Applies to BOTH query and document!)
    "NorQuadRetrieval": "task: question answering | query: ",
    "SweFaqRetrieval": "task: question answering | query: ",
    "TwitterHjerneRetrieval": "task: question answering | query: ",
}


saga_embed_v1 = ModelMeta(
    name="nicher92/saga-embed_v1",
    reference="https://huggingface.co/nicher92/saga-embed_v1",
    revision="3be07ac3d7c3e00e4402ae9285b23fcf8fda6735",
    release_date="2025-01-09",
    languages=["swe-Latn"],
    n_parameters=404_219_904,
    memory_usage_mb=2167,
    license="mit",
    max_tokens=1024,
    embed_dim=1024,
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    similarity_fn_name="cosine",
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "RedditClustering",
        # "sentence-transformers/xsum",
        # "sentence-transformers/simple-wiki",
        # "sentence-transformers/s2orc",
        # "sentence-transformers/amazon-reviews",
        # "sentence-transformers/gooaq",
        # "sentence-transformers/paq",
        "StackExchangeClustering",
        # "sentence-transformers/wikipedia-sections",
        # "stanfordnlp/snli",
        "NQ",
    },
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={"model_prompts": saga_prompts}
)
