"""Implementation of Sentence Transformers model validated in MTEB."""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
    sentence_transformers_loader,
)

paraphrase_langs = [
    "ara-Arab",
    "bul-Cyrl",
    "cat-Latn",
    "ces-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "kat-Geor",
    "kor-Hang",
    "kur-Arab",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mya-Mymr",
    "nob-Latn",
    "nld-Latn",
    "pol-Latn",
    "por-Latn",
    "por-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "slk-Latn",
    "slv-Latn",
    "sqi-Latn",
    "srp-Cyrl",
    "swe-Latn",
    "tha-Thai",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "zho-Hans",
    "zho-Hant",
]

SBERT_CITATION = """@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
"""


sent_trf_training_dataset = {
    # derived from datasheets
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
    "MSMARCO-PL",  # translation not trained on
    "mMARCO-NL",  # translation not trained on
    "NQ",
    "NQHardNegatives",
    "NanoNQRetrieval",
    "NQ-PL",  # translation not trained on
    "NQ-NL",  # translation not trained on
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

all_minilm_l6_v2 = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/all-MiniLM-L6-v2",
    languages=["eng-Latn"],
    open_weights=True,
    revision="8b3219a92973c328a8e22fadcfa821b5dc75636a",
    release_date="2021-08-30",
    n_parameters=22_700_000,
    memory_usage_mb=87,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=256,
    reference="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=sent_trf_training_dataset,
    public_training_code=None,
    public_training_data=None,
    citation=SBERT_CITATION,
)

all_minilm_l12_v2 = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/all-MiniLM-L12-v2",
    languages=["eng-Latn"],
    open_weights=True,
    revision="364dd28d28dcd3359b537f3cf1f5348ba679da62",
    release_date="2021-08-30",
    n_parameters=33_400_000,
    memory_usage_mb=127,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=256,
    reference="https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=sent_trf_training_dataset,
    public_training_code=None,
    citation=SBERT_CITATION,
    public_training_data=None,
)

paraphrase_multilingual_minilm_l12_v2 = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    languages=paraphrase_langs,
    open_weights=True,
    revision="bf3bf13ab40c3157080a7ab344c831b9ad18b5eb",
    release_date="2019-11-01",  # release date of paper
    n_parameters=118_000_000,
    memory_usage_mb=449,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=sent_trf_training_dataset,  # assumed (probably some parallel as well)
    public_training_code=None,
    citation=SBERT_CITATION,
    public_training_data=None,
)

paraphrase_multilingual_mpnet_base_v2 = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    languages=paraphrase_langs,
    open_weights=True,
    revision="79f2382ceacceacdf38563d7c5d16b9ff8d725d6",
    release_date="2019-11-01",  # release date of paper
    n_parameters=278_000_000,
    memory_usage_mb=1061,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    citation=SBERT_CITATION,
    training_datasets=sent_trf_training_dataset,
    # + https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/paraphrases/training.py
    # which include (not in MTEB):
    # "all-nli": all_nli_train_dataset,
    # "sentence-compression": sentence_compression_train_dataset,
    # "simple-wiki": simple_wiki_train_dataset,
    # "altlex": altlex_train_dataset,
    # "quora-duplicates": quora_train_dataset,
    # "coco-captions": coco_train_dataset,
    # "flickr30k-captions": flickr_train_dataset,
    # "yahoo-answers": yahoo_answers_train_dataset,
    # "stack-exchange": stack_exchange_train_dataset,
    public_training_code=None,
    public_training_data=None,
)

labse = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/LaBSE",
    languages=paraphrase_langs,
    open_weights=True,
    revision="e34fab64a3011d2176c99545a93d5cbddc9a91b7",
    release_date="2019-11-01",  # release date of paper
    n_parameters=471_000_000,
    memory_usage_mb=1796,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/LaBSE",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=set(
        # CommonCrawl
        # wiki  05-21-2020 dump
        # The translation corpus is constructed from web pages using a bitext mining system
    ),
    # scraped and mined webdata including CC, wiki, see section 3.1 https://aclanthology.org/2022.acl-long.62.pdf
    public_training_code="https://www.kaggle.com/models/google/labse/tensorFlow2/labse/2?tfhub-redirect=true",
    citation="""@misc{feng2022languageagnosticbertsentenceembedding,
      title={Language-agnostic BERT Sentence Embedding},
      author={Fangxiaoyu Feng and Yinfei Yang and Daniel Cer and Naveen Arivazhagan and Wei Wang},
      year={2022},
      eprint={2007.01852},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2007.01852},
    }""",
    public_training_data=None,
)

multi_qa_minilm_l6_cos_v1 = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b207367332321f8e44f96e224ef15bc607f4dbf0",
    release_date="2021-08-30",
    n_parameters=22_700_000,
    memory_usage_mb=87,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from="nreimers/MiniLM-L6-H384-uncased",
    training_datasets=sent_trf_training_dataset,  # assumed
    public_training_code=None,
    public_training_data=None,
    citation=SBERT_CITATION,
)

all_mpnet_base_v2 = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/all-mpnet-base-v2",
    languages=["eng-Latn"],
    open_weights=True,
    revision="9a3225965996d404b775526de6dbfe85d3368642",
    release_date="2021-08-30",
    n_parameters=109_000_000,
    memory_usage_mb=418,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=384,
    reference="https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=sent_trf_training_dataset,
    public_training_code=None,
    public_training_data=None,
    citation=SBERT_CITATION,
)

static_multi_datasets = {
    # wikititles
    "TatoebaBitextMining",
    # talks
    # europarl
    # global_voices
    # muse
    # wikimatrix
    # opensubtitles
    "StackExchangeClustering.v2",
    "StackExchangeClusteringP2P.v2",
    # quora
    "QuoraRetrieval",
    "NanoQuoraRetrieval",
    "Quora-NL",  # translation not trained on
    # wikianswers_duplicates
    # all_nli - the "triplet" subset
    # simple_wiki
    # altlex
    # flickr30k_captions
    # coco_captions
    # nli_for_simcse
    # negation
}
static_multi_languages = [
    "eng-Latn",
    "ara-Arab",
    "bul-Cyrl",
    "cat-Latn",
    "ces-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "spa-Latn",
    "est-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "kat-Geor",
    "kor-Hang",
    "kur-Latn",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mon-Cyrl",
    "mar-Deva",
    "mal-Mlym",
    "mya-Mymr",
    "nob-Latn",
    "nld-Latn",
    "pol-Latn",
    "por-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "slk-Latn",
    "slv-Latn",
    "sqi-Latn",
    "srp-Cyrl",
    "swe-Latn",
    "tha-Thai",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "zho-Hans",
]

static_similarity_mrl_multilingual_v1 = ModelMeta(
    name="sentence-transformers/static-similarity-mrl-multilingual-v1",
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(
        device="cpu",  # CPU is just as quick, if not quicker
    ),
    languages=static_multi_languages,
    open_weights=True,
    revision="7264ea07c5365a11d7e6d87dbb6195889a13054f",
    release_date="2025-01-15",
    n_parameters=108_420_096,
    memory_usage_mb=413,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=None,
    reference="https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=static_multi_datasets,
    public_training_code="https://huggingface.co/blog/static-embeddings",
    public_training_data="https://huggingface.co/collections/sentence-transformers/embedding-model-datasets-6644d7a3673a511914aa7552",
)

contriever = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="facebook/contriever-msmarco",
    languages=["eng-Latn"],
    open_weights=True,
    revision="abe8c1493371369031bcb1e02acb754cf4e162fa",
    release_date="2022-06-25",  # release date of model on HF
    n_parameters=150_000_000,
    memory_usage_mb=572,
    embed_dim=768,
    license=None,
    max_tokens=512,
    reference="https://huggingface.co/facebook/contriever-msmarco",
    similarity_fn_name=ScoringFunction.DOT_PRODUCT,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    citation="""
    @misc{izacard2021contriever,
      title={Unsupervised Dense Information Retrieval with Contrastive Learning},
      author={Gautier Izacard and Mathilde Caron and Lucas Hosseini and Sebastian Riedel and Piotr Bojanowski and Armand Joulin and Edouard Grave},
      year={2021},
      url = {https://arxiv.org/abs/2112.09118},
      doi = {10.48550/ARXIV.2112.09118},
    }""",
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

microllama_text_embedding = ModelMeta(
    loader=sentence_transformers_loader,
    name="keeeeenw/MicroLlama-text-embedding",
    languages=["eng-Latn"],
    open_weights=True,
    revision="98f70f14cdf12d7ea217ed2fd4e808b0195f1e7e",
    release_date="2024-11-10",
    n_parameters=272_000_000,
    memory_usage_mb=1037,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=2048,
    reference="https://huggingface.co/keeeeenw/MicroLlama-text-embedding",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets={
        "NQ",
        "NQHardNegatives",
        "NanoNQRetrieval",
        "NQ-PL",  # translation not trained on
        "NQ-NL",  # translation not trained on
        # not in MTEB
        # "sentence-transformers/all-nli",
        # "sentence-transformers/stsb",
        # "sentence-transformers/quora-duplicates",
        # "sentence-transformers/natural-questions",
    },
    public_training_code=None,
    public_training_data=None,
)

sentence_t5_base = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/sentence-t5-base",
    languages=["eng-Latn"],
    open_weights=True,
    revision="50c53e206f8b01c9621484a3c0aafce4e55efebf",
    release_date="2022-02-09",
    n_parameters=110_000_000,
    memory_usage_mb=209,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/sentence-t5-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"SNLI", "Community QA"},
)

sentence_t5_large = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/sentence-t5-large",
    languages=["eng-Latn"],
    open_weights=True,
    revision="1fc08ea477205aa54a3e5b13f0971ae16b86410a",
    release_date="2022-02-09",
    n_parameters=335_000_000,
    memory_usage_mb=639,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/sentence-t5-large",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"SNLI", "Community QA"},
)

sentence_t5_xl = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/sentence-t5-xl",
    languages=["eng-Latn"],
    open_weights=True,
    revision="2965d31b368fb14117688e0bde77cbd720e91f53",
    release_date="2024-03-27",
    n_parameters=3_000_000_000,
    memory_usage_mb=2367,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/sentence-t5-xl",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"SNLI", "Community QA"},
)

sentence_t5_xxl = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/sentence-t5-xxl",
    languages=["eng-Latn"],
    open_weights=True,
    revision="4d122282ba80e807e9e6eb8c358269e92796365d",
    release_date="2024-03-27",
    n_parameters=11_000_000_000,
    memory_usage_mb=9279,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/sentence-t5-xxl",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"SNLI", "Community QA"},
)
gtr_t5_large = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/gtr-t5-large",
    languages=["eng-Latn"],  # in format eng-Latn
    open_weights=True,
    revision="a2c8ac47f998531948d4cbe32a0b577a7037a5e3",
    release_date="2022-02-09",
    n_parameters=335_000_000,
    memory_usage_mb=639,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/gtr-t5-large",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
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
        "Community QA",
    },
)

gtr_t5_xl = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/gtr-t5-xl",
    languages=["eng-Latn"],  # in format eng-Latn
    open_weights=True,
    revision="23a8d667a1ad2578af181ce762867003c498d1bf",
    release_date="2022-02-09",
    n_parameters=1_240_000_000,
    memory_usage_mb=2367,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/gtr-t5-xl",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
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
        "Community QA",
    },
)
gtr_t5_xxl = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/gtr-t5-xxl",
    languages=["eng-Latn"],  # in format eng-Latn
    open_weights=True,
    revision="73f2a9156a3dcc2194dfdb2bf201cd7d17e17884",
    release_date="2022-02-09",
    n_parameters=4_860_000_000,
    memory_usage_mb=9279,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/gtr-t5-xxl",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
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
        "Community QA",
    },
)

gtr_t5_base = ModelMeta(
    loader=sentence_transformers_loader,
    name="sentence-transformers/gtr-t5-base",
    languages=["eng-Latn"],  # in format eng-Latn
    open_weights=True,
    revision="7027e9594267928589816394bdd295273ddc0739",
    release_date="2022-02-09",
    n_parameters=110_000_000,
    memory_usage_mb=209,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/gtr-t5-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
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
        "Community QA",
    },
)
