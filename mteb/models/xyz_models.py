from __future__ import annotations

from mteb.model_meta import ModelMeta, sentence_transformers_loader

xyz_zh_datasets = {
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
}
# not in mteb
# "Covid-News":"NCPPolicies_train",
# "cMedQA":"https://github.com/zhangsheng93/cMedQA",
# "Multi-CPR":"http://github.com/Alibaba-NLP/Multi-CPR",
# "retrieval_data_llm":"https://huggingface.co/datasets/infgrad/retrieval_data_llm",
# "Huatuo26M-Lite":"https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite"

xyz_embedding = ModelMeta(
    name="fangxq/XYZ-embedding",
    languages=["zho-Hans"],
    loader=sentence_transformers_loader,
    open_weights=True,
    revision="4004120220b99baea764a1d3508427248ac3bccf",
    release_date="2024-09-13",
    n_parameters=326000000,
    memory_usage_mb=1242,
    max_tokens=512,
    embed_dim=768,
    license="mit",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers"],
    reference="https://huggingface.co/fangxq/XYZ-embedding",
    use_instructions=False,
    training_datasets=xyz_zh_datasets,
    public_training_code=None,
    public_training_data=None,
)
