"""Implementation of Sentence Transformers model validated in MTEB."""

from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper

paraphrase_langs = [
    "ara_Arab",
    "bul_Cyrl",
    "cat_Latn",
    "ces_Latn",
    "dan_Latn",
    "deu_Latn",
    "ell_Grek",
    "eng_Latn",
    "spa_Latn",
    "est_Latn",
    "fas_Arab",
    "fin_Latn",
    "fra_Latn",
    "fra_Latn",
    "glg_Latn",
    "guj_Gujr",
    "heb_Hebr",
    "hin_Deva",
    "hrv_Latn",
    "hun_Latn",
    "hye_Armn",
    "ind_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "kat_Geor",
    "kor_Hang",
    "kur_Arab",
    "lit_Latn",
    "lav_Latn",
    "mkd_Cyrl",
    "mon_Cyrl",
    "mar_Deva",
    "msa_Latn",
    "mya_Mymr",
    "nob_Latn",
    "nld_Latn",
    "pol_Latn",
    "por_Latn",
    "por_Latn",
    "ron_Latn",
    "rus_Cyrl",
    "slk_Latn",
    "slv_Latn",
    "sqi_Latn",
    "srp_Cyrl",
    "swe_Latn",
    "tha_Thai",
    "tur_Latn",
    "ukr_Cyrl",
    "urd_Arab",
    "vie_Latn",
    "zho_Hans",
    "zho_Hant",
]

sent_trf_training_dataset = {
    # derived from datasheets
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "MSMARCO-PL": ["train"],  # translation not trained on
    "mMARCO-NL": ["train"],  # translation not trained on
    "NQ": ["train"],
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    "NQ-PL": ["train"],  # translation not trained on
    "NQ-NL": ["train"],  # translation not trained on
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

all_MiniLM_L6_v2 = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=sent_trf_training_dataset,
    public_training_code=None,
    public_training_data=None,
)

all_MiniLM_L12_v2 = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=sent_trf_training_dataset,
    public_training_code=None,
    public_training_data=None,
)

paraphrase_multilingual_MiniLM_L12_v2 = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=sent_trf_training_dataset,  # assumed (probably some parallel as well)
    public_training_code=None,
    public_training_data=None,
)

paraphrase_multilingual_mpnet_base_v2 = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets={
        # CommonCrawl
        # wiki  05-21-2020 dump
        # The translation corpus is constructed from web pages using a bitext mining system
    },
    # scraped and mined webdata including CC, wiki, see section 3.1 https://aclanthology.org/2022.acl-long.62.pdf
    public_training_code="https://www.kaggle.com/models/google/labse/tensorFlow2/labse/2?tfhub-redirect=true",
    public_training_data=None,
)

multi_qa_MiniLM_L6_cos_v1 = ModelMeta(
    name="sentence-transformer/multi-qa-MiniLM-L6-cos-v1",
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from="nreimers/MiniLM-L6-H384-uncased",
    training_datasets=sent_trf_training_dataset,  # assumed
    public_training_code=None,
    public_training_data=None,
)

all_mpnet_base_v2 = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=sent_trf_training_dataset,
    public_training_code=None,
    public_training_data=None,
)

static_multi_datasets = {
    # wikititles
    "TatoebaBitextMining": [],
    # talks
    # europarl
    # global_voices
    # muse
    # wikimatrix
    # opensubtitles
    "StackExchangeClustering.v2": [],
    "StackExchangeClusteringP2P.v2": [],
    # quora
    "QuoraRetrieval": [],
    "NanoQuoraRetrieval": [],
    "Quora-NL": [],  # translation not trained on
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
    "eng_Latn",
    "ara_Arab",
    "bul_Cyrl",
    "cat_Latn",
    "ces_Latn",
    "dan_Latn",
    "deu_Latn",
    "ell_Grek",
    "spa_Latn",
    "est_Latn",
    "fas_Arab",
    "fin_Latn",
    "fra_Latn",
    "glg_Latn",
    "guj_Gujr",
    "heb_Hebr",
    "hin_Deva",
    "hun_Latn",
    "hye_Armn",
    "ind_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "kat_Geor",
    "kor_Hang",
    "kur_Latn",
    "lit_Latn",
    "lav_Latn",
    "mkd_Cyrl",
    "mon_Cyrl",
    "mar_Deva",
    "mal_Mlym",
    "mya_Mymr",
    "nob_Latn",
    "nld_Latn",
    "pol_Latn",
    "por_Latn",
    "ron_Latn",
    "rus_Cyrl",
    "slk_Latn",
    "slv_Latn",
    "sqi_Latn",
    "srp_Cyrl",
    "swe_Latn",
    "tha_Thai",
    "tur_Latn",
    "ukr_Cyrl",
    "urd_Arab",
    "vie_Latn",
    "zho_Hans",
]

static_similarity_mrl_multilingual_v1 = ModelMeta(
    name="sentence-transformers/static-similarity-mrl-multilingual-v1",
    loader=partial(
        SentenceTransformerWrapper,
        model="sentence-transformers/static-similarity-mrl-multilingual-v1",
        revision="7264ea07c5365a11d7e6d87dbb6195889a13054f",
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

static_eng_datasets = {
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "MSMARCO-PL": ["train"],  # translation not trained on
    "mMARCO-NL": ["train"],  # translation not trained on
    # gooaq
    # s2orc
    # squad
    # all-nli (multi-nli, snli)
    # paq
    # trivia-qa
    # msmarco-10m
    # swim_ir
    # pubmedqa
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
    "MIRACLReranking": ["train"],
    "MultiLongDocRetrieval": ["train"],
    "MrTidyRetrieval": ["train"],
}

static_retrieval_mrl_en_v1 = ModelMeta(
    name="sentence-transformers/static-retrieval-mrl-en-v1",
    loader=partial(
        SentenceTransformerWrapper,
        model="sentence-transformers/static-retrieval-mrl-en-v1",
        revision="f60985c706f192d45d218078e49e5a8b6f15283a",
        device="cpu",  # CPU is extremely quick already
    ),
    languages=["eng-Latn"],
    open_weights=True,
    revision="f60985c706f192d45d218078e49e5a8b6f15283a",
    release_date="2025-01-15",
    n_parameters=31_254_528,
    memory_usage_mb=119,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=None,
    reference="https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets=static_eng_datasets,
    public_training_code="https://huggingface.co/blog/static-embeddings",
    public_training_data="https://huggingface.co/collections/sentence-transformers/embedding-model-datasets-6644d7a3673a511914aa7552",
)

microllama_text_embedding = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets={
        "NQ": ["train"],
        "NQHardNegatives": ["train"],
        "NanoNQRetrieval": ["train"],
        "NQ-PL": ["train"],  # translation not trained on
        "NQ-NL": ["train"],  # translation not trained on
        # not in MTEB
        # "sentence-transformers/all-nli": ["train"],
        # "sentence-transformers/stsb": ["train"],
        # "sentence-transformers/quora-duplicates": ["train"],
        # "sentence-transformers/natural-questions": ["train"],
    },
    public_training_code=None,
    public_training_data=None,
)

sentence_t5_base = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"SNLI": ["train"], "Community QA": ["train"]},
)

sentence_t5_large = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"SNLI": ["train"], "Community QA": ["train"]},
)

sentence_t5_xl = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"SNLI": ["train"], "Community QA": ["train"]},
)

sentence_t5_xxl = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={"SNLI": ["train"], "Community QA": ["train"]},
)
gtr_t5_large = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
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
        "Community QA": ["train"],
    },
)

gtr_t5_xl = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
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
        "Community QA": ["train"],
    },
)
gtr_t5_xxl = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
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
        "Community QA": ["train"],
    },
)

gtr_t5_base = ModelMeta(
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
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
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
        "Community QA": ["train"],
    },
)
