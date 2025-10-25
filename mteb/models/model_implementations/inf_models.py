from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

inf_retriever_v1_training_data = {
    # eng_Latn
    "ArguAna",
    "CQADupstackRetrieval",
    "ClimateFEVER",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "TRECCOVID",
    "Touche2020",
    # and other private data of INF TECH (not in MTEB),
    #
    # zho_Hans
    "CmedqaRetrieval",
    "CovidRetrieval",
    "DuRetrieval",
    "EcomRetrieval",
    "MMarcoRetrieval",
    "MedicalRetrieval",
    "T2Retrieval",
    "VideoRetrieval",
    # and other private data of INF TECH (not in MTEB),
}

INF_RETRIEVER_CITATION = """@misc{infly-ai_2025,
  author       = {Junhan Yang and Jiahe Wan and Yichen Yao and Wei Chu and Yinghui Xu and Yuan Qi},
  title        = {inf-retriever-v1 (Revision 5f469d7)},
  year         = 2025,
  url          = {https://huggingface.co/infly/inf-retriever-v1},
  doi          = {10.57967/hf/4262},
  publisher    = {Hugging Face}
}"""

inf_retriever_v1 = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="infly/inf-retriever-v1",
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    revision="cb70ca7c31dfa866b2eff2dad229c144d8ddfd91",
    release_date="2024-12-24",  # initial commit of hf model.
    n_parameters=7_069_121_024,
    memory_usage_mb=13483,
    embed_dim=3584,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/infly/inf-retriever-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="Alibaba-NLP/gte-Qwen2-7B-instruct",
    public_training_code=None,
    public_training_data=None,
    training_datasets=inf_retriever_v1_training_data,
    citation=INF_RETRIEVER_CITATION,
)

inf_retriever_v1_1_5b = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
    name="infly/inf-retriever-v1-1.5b",
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    revision="c9c05c2dd50707a486966ba81703021ae2094a06",
    release_date="2025-02-08",  # initial commit of hf model.
    n_parameters=1_543_268_864,
    memory_usage_mb=2944,
    embed_dim=1536,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/infly/inf-retriever-v1-1.5b",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    public_training_code=None,
    public_training_data=None,
    training_datasets=inf_retriever_v1_training_data,
    citation=INF_RETRIEVER_CITATION,
)
