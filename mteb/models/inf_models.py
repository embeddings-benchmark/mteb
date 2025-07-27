from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

inf_retreiver_v1_training_data = {
    # eng_Latn
    "ArguAna": ["train"],
    "CQADupstackRetrieval": ["train"],
    "ClimateFEVER": ["train"],
    "DBPedia": ["train"],
    "FEVER": ["train"],
    "FiQA2018": ["train"],
    "HotpotQA": ["train"],
    "MSMARCO": ["train"],
    "NFCorpus": ["train"],
    "NQ": ["train"],
    "QuoraRetrieval": ["train"],
    "SCIDOCS": ["train"],
    "SciFact": ["train"],
    "TRECCOVID": ["train"],
    "Touche2020": ["train"],
    ## and other private data of INF TECH (not in MTEB),
    #
    # zho_Hans
    "CmedqaRetrieval": ["train"],
    "CovidRetrieval": ["train"],
    "DuRetrieval": ["train"],
    "EcomRetrieval": ["train"],
    "MMarcoRetrieval": ["train"],
    "MedicalRetrieval": ["train"],
    "T2Retrieval": ["train"],
    "VideoRetrieval": ["train"],
    ## and other private data of INF TECH (not in MTEB),
}

inf_retriever_v1 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="infly/inf-retriever-v1",
        revision="cb70ca7c31dfa866b2eff2dad229c144d8ddfd91",
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="Alibaba-NLP/gte-Qwen2-7B-instruct",
    public_training_code=None,
    public_training_data=None,
    training_datasets=inf_retreiver_v1_training_data,
)

inf_retriever_v1_1_5B = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="infly/inf-retriever-v1-1.5b",
        revision="c9c05c2dd50707a486966ba81703021ae2094a06",
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    public_training_code=None,
    public_training_data=None,
    training_datasets=inf_retreiver_v1_training_data,
)
