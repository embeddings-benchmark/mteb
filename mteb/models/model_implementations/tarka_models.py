from mteb.models.model_meta import ModelMeta
from mteb.models.model_implementations.google_models import gemma_embedding_loader

Tarka_Embedding_150M_V1_CITATION = '''@misc{tarka_ai_research_2025,
	author       = { Tarka AI Research },
	title        = { Tarka-Embedding-150M-V1 (Revision c5f4f43) },
	year         = 2025,
	url          = { https://huggingface.co/Tarka-AIR/Tarka-Embedding-150M-V1 },
	doi          = { 10.57967/hf/6875 },
	publisher    = { Hugging Face }
}'''

MULTILINGUAL_EVALUATED_LANGUAGES = [
    "eng-Latn",
]


training_data = {
    "T2Retrieval",
    "DuRetrieval",
    "MMarcoReranking",
    "CMedQAv2-reranking",
    "NQ",
    "MSMARCO",
    "HotpotQA",
    "FEVER",
    "MrTidyRetrieval",
    "MIRACLRetrieval",
    "CodeSearchNet",
}

tarka_embedding_150m_v1 = ModelMeta(
    loader=gemma_embedding_loader,
    name="Tarka-AIR/Tarka-Embedding-150M-V1",
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="c5f4f43",
    release_date="2025-11-04",
    n_parameters=155_714_304,
    embed_dim=768,
    max_tokens=2048,
    license="gemma",
    reference="https://ai.google.dev/gemma/docs/embeddinggemma/model_card",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    similarity_fn_name="cosine",
    memory_usage_mb=576,
    citation=Tarka_Embedding_150M_V1_CITATION,
)