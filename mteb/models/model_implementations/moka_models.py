"""Moka AI's Chinese embedding models"""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

M3E_CITATION = """@software{MokaMassiveMixedEmbedding,
  author = {Wang Yuxin and Sun Qingxuan and He Sicheng},
  title = {M3E: Moka Massive Mixed Embedding Model},
  year = {2023}
}"""

sent_trf_training_dataset = {
    # derived from datasheets
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
    "MSMARCO-PL",  # translation not trained on
    "mMARCO-NL",  # translation not trained on
    "NQ",
    "NQ-NL",  # translation not trained on
    "NQHardNegatives",
    "NanoNQRetrieval",
    "NQ-PL",  # translation not trained on
    # not in MTEB
    # "s2orc",
    # "flax-sentence-embeddings/stackexchange_xml",
    # "ms_marco",
    # "gooaq",
    # "yahoo_answers_topics",
    # "code_search_net",
    # "search_qa",
    # "eli5",
    # "snli",
    # "multi_nli",
    # "wikihow",
    # "natural_questions",
    # "trivia_qa",
    # "embedding-data/sentence-compression",
    # "embedding-data/flickr30k-captions",
    # "embedding-data/altlex",
    # "embedding-data/simple-wiki",
    # "embedding-data/QQP",
    # "embedding-data/SPECTER",
    # "embedding-data/PAQ_pairs",
    # "embedding-data/WikiAnswers",
}
medi_dataset = (
    set(
        # not in MTEB:
        # - Super-NI
        # - KILT (https://arxiv.org/abs/2009.02252)
        # - MedMCQA (https://proceedings.mlr.press/v174/pal22a/pal22a.pdf)
    )
    | sent_trf_training_dataset
)
m3e_dataset = {
    "AmazonReviewsClassification",  # Possibly also test, hard to know
    "Ocnli",
    "BQ",
    "LCQMC",
    "MIRACLReranking",
    "PAWSX",
    "DuRetrieval",
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
} | medi_dataset

m3e_base = ModelMeta(
    loader=sentence_transformers_loader,
    name="moka-ai/m3e-base",
    languages=["zho-Hans", "eng-Latn"],
    open_weights=True,
    revision="764b537a0e50e5c7d64db883f2d2e051cbe3c64c",
    release_date="2023-06-06",  # first commit
    n_parameters=int(102 * 1e6),
    memory_usage_mb=390,
    embed_dim=768,
    # They don't give a specific license but commercial use is not allowed
    license="https://huggingface.co/moka-ai/m3e-base#%F0%9F%93%9C-license",
    max_tokens=512,
    reference="https://huggingface.co/moka-ai/m3e-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,  # Not published
    training_datasets=m3e_dataset,
    citation=M3E_CITATION,
)

m3e_small = ModelMeta(
    loader=sentence_transformers_loader,
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
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,  # Not published
    training_datasets=m3e_dataset,
    citation=M3E_CITATION,
)

m3e_large = ModelMeta(
    loader=sentence_transformers_loader,
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
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,  # Not published
    training_datasets=m3e_dataset,
    citation=M3E_CITATION,
)
