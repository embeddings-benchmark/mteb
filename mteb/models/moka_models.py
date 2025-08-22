"""Moka AI's Chinese embedding models"""

from __future__ import annotations

from mteb.model_meta import ModelMeta

sent_trf_training_dataset = {
    # derived from datasheets
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "MSMARCO-PL": ["train"],  # translation not trained on
    "mMARCO-NL": ["train"],  # translation not trained on
    "NQ": ["train"],
    "NQ-NL": ["train"],  # translation not trained on
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    "NQ-PL": ["train"],  # translation not trained on
    # not in MTEB
    # "s2orc": ["train"],
    # "flax-sentence-embeddings/stackexchange_xml": ["train"],
    # "ms_marco": ["train"],
    # "gooaq": ["train"],
    # "yahoo_answers_topics": ["train"],
    # "code_search_net": ["train"],
    # "search_qa": ["train"],
    # "eli5": ["train"],
    # "snli": ["train"],
    # "multi_nli": ["train"],
    # "wikihow": ["train"],
    # "natural_questions": ["train"],
    # "trivia_qa": ["train"],
    # "embedding-data/sentence-compression": ["train"],
    # "embedding-data/flickr30k-captions": ["train"],
    # "embedding-data/altlex": ["train"],
    # "embedding-data/simple-wiki": ["train"],
    # "embedding-data/QQP": ["train"],
    # "embedding-data/SPECTER": ["train"],
    # "embedding-data/PAQ_pairs": ["train"],
    # "embedding-data/WikiAnswers": ["train"],
}
medi_dataset = {
    **sent_trf_training_dataset,
    # not in MTEB:
    # - Super-NI
    # - KILT (https://arxiv.org/abs/2009.02252)
    # - MedMCQA (https://proceedings.mlr.press/v174/pal22a/pal22a.pdf)
}
m3e_dataset = {
    **medi_dataset,
    "AmazonReviewsClassification": ["train"],  # Possibly also test, hard to know
    "Ocnli": ["train"],
    "BQ": ["train"],
    "LCQMC": ["train"],
    "MIRACLReranking": ["train"],
    "PAWSX": ["train"],
    "DuRetrieval": [],
    # not in MTEB:
    # - cmrc2018
    # - belle_2m
    # - firefily
    # - alpaca_gpt4
    # - zhihu_kol
    # - hc3_chinese
    # - amazon_reviews_multi (intersects with AmazonReviewsClassification)
    # - qa: Encyclopedia QA dataset
    # - xlsum
    # - wiki_atomic_edit
    # - chatmed_consult
    # - webqa
    # - dureader_robust - DuRetrieval
    # - csl
    # - lawzhidao
    # - CINLID
    # - DuSQL
    # - Zhuiyi-NL2SQL
    # - Cspider
    # - news2016zh
    # - baike2018qa
    # - webtext2019zh
    # - SimCLUE
    # - SQuAD
}

m3e_base = ModelMeta(
    name="moka-ai/m3e-base",
    languages=["zho-Hans", "eng-Latn"],
    open_weights=True,
    revision="764b537a0e50e5c7d64db883f2d2e051cbe3c64c",
    release_date="2023-06-06",  # first commit
    n_parameters=102 * 1e6,
    memory_usage_mb=390,
    embed_dim=768,
    # They don't give a specific license but commercial use is not allowed
    license="https://huggingface.co/moka-ai/m3e-base#%F0%9F%93%9C-license",
    max_tokens=512,
    reference="https://huggingface.co/moka-ai/m3e-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,  # Not published
    training_datasets=m3e_dataset,
)

m3e_small = ModelMeta(
    name="moka-ai/m3e-small",
    languages=["zho-Hans", "eng-Latn"],
    open_weights=True,
    revision="44c696631b2a8c200220aaaad5f987f096e986df",
    release_date="2023-06-02",  # first commit
    n_parameters=None,
    memory_usage_mb=None,  # Can't be seen on HF page
    embed_dim=512,
    # They don't give a specific license but commercial use is not allowed
    license="https://huggingface.co/moka-ai/m3e-base#%F0%9F%93%9C-license",
    max_tokens=512,
    reference="https://huggingface.co/moka-ai/m3e-small",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,  # Not published
    training_datasets=m3e_dataset,
)

m3e_large = ModelMeta(
    name="moka-ai/m3e-large",
    languages=["zho-Hans", "eng-Latn"],
    open_weights=True,
    revision="12900375086c37ba5d83d1e417b21dc7d1d1f388",
    release_date="2023-06-21",  # first commit
    n_parameters=None,
    memory_usage_mb=None,  # Can't be seen on HF page
    embed_dim=768,
    # They don't give a specific license but commercial use is not allowed
    license="https://huggingface.co/moka-ai/m3e-base#%F0%9F%93%9C-license",
    max_tokens=512,
    reference="https://huggingface.co/moka-ai/m3e-large",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,  # Not published
    training_datasets=m3e_dataset,
)
