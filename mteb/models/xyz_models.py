from __future__ import annotations

from mteb.model_meta import ModelMeta, sentence_transformers_loader

xyz_embedding = ModelMeta(
    name="fangxq/XYZ-embedding",
    languages=["zho-Hans"],
    loader=sentence_transformers_loader,
    open_weights=False,
    revision="4004120220b99baea764a1d3508427248ac3bccf",
    release_date="2024-09-13",
    n_parameters=326000000,
    memory_usage_mb=1242,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/fangxq/XYZ-embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
    ##
    "BQ": ["train"],
    "LCQMC": ["train"],
    "PAWSX": ["train"],
    "STS-B": ["train"],
    "DuRetrieval": ["train"],
    "AFQMC": ["train"],
    "Cmnli": ["train"],
    "Ocnli": ["train"],
    "T2Retrieval": ["train"],
    "T2Reranking": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "Multi-CPR":"http://github.com/Alibaba-NLP/Multi-CPR"},)
