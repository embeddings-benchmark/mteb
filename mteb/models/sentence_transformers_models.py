"""Implementation of Sentence Transformers model validated in MTEB."""

from __future__ import annotations

from mteb.model_meta import ModelMeta

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

all_MiniLM_L6_v2 = ModelMeta(
    name="sentence-transformers/all-MiniLM-L6-v2",
    languages=["eng-Latn"],
    open_weights=True,
    revision="8b3219a92973c328a8e22fadcfa821b5dc75636a",
    release_date="2021-08-30",
    n_parameters=22_700_000,
    memory_usage=None,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=256,
    reference="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=False,  # does sentence transformer count?
    public_training_data=True,
    training_datasets={
        # source: frontmatter in readme
        # trained on stack exchange, unsure if sources match
        "StackExchangeClusteringP2P": ["test"],
        "StackExchangeClusteringP2P.v2": ["test"],
        "StackExchangeClustering": ["test"],
        "StackExchangeClustering.v2": ["test"],
        "NQ": ["test"],
        "NQHardNegatives": ["test"],
        "MSMARCO": ["train"],
        # Non MTEB sources
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
        # "trivia_qa": ["train"],
        # "embedding-data/sentence-compression": ["train"],
        # "embedding-data/flickr30k-captions": ["train"],
        # "embedding-data/altlex": ["train"],
        # "embedding-data/simple-wiki": ["train"],
        # "embedding-data/QQP": ["train"],
        # "embedding-data/SPECTER": ["train"],
        # "embedding-data/PAQ_pairs": ["train"],
        # "embedding-data/WikiAnswers": ["train"],
    },
)

paraphrase_multilingual_MiniLM_L12_v2 = ModelMeta(
    name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    languages=paraphrase_langs,
    open_weights=True,
    revision="bf3bf13ab40c3157080a7ab344c831b9ad18b5eb",
    release_date="2019-11-01",  # release date of paper
    n_parameters=118_000_000,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
)

paraphrase_multilingual_mpnet_base_v2 = ModelMeta(
    name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    languages=paraphrase_langs,
    open_weights=True,
    revision="79f2382ceacceacdf38563d7c5d16b9ff8d725d6",
    release_date="2019-11-01",  # release date of paper
    n_parameters=278_000_000,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
)

labse = ModelMeta(
    name="sentence-transformers/LaBSE",
    languages=paraphrase_langs,
    open_weights=True,
    revision="e34fab64a3011d2176c99545a93d5cbddc9a91b7",
    release_date="2019-11-01",  # release date of paper
    n_parameters=471_000_000,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/LaBSE",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
)

multi_qa_MiniLM_L6_cos_v1 = ModelMeta(
    name="sentence-transformer/multi-qa-MiniLM-L6-cos-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b207367332321f8e44f96e224ef15bc607f4dbf0",
    release_date="2021-08-30",
    n_parameters=22_700_000,
    memory_usage=None,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
)

all_mpnet_base_v2 = ModelMeta(
    name="sentence-transformers/all-mpnet-base-v2",
    languages=["eng-Latn"],
    open_weights=True,
    revision="9a3225965996d404b775526de6dbfe85d3368642",
    release_date="2021-08-30",
    n_parameters=109_000_000,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=384,
    reference="https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=False,  # does sentence transformer count?
    public_training_data=True,
    training_datasets={
        # source: frontmatter in readme
        # trained on stack exchange, unsure if sources match
        "StackExchangeClusteringP2P": ["test"],
        "StackExchangeClusteringP2P.v2": ["test"],
        "StackExchangeClustering": ["test"],
        "StackExchangeClustering.v2": ["test"],
        "NQ": ["test"],
        "NQHardNegatives": ["test"],
        "MSMARCO": ["train"],
        # Non MTEB sources
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
        # "trivia_qa": ["train"],
        # "embedding-data/sentence-compression": ["train"],
        # "embedding-data/flickr30k-captions": ["train"],
        # "embedding-data/altlex": ["train"],
        # "embedding-data/simple-wiki": ["train"],
        # "embedding-data/QQP": ["train"],
        # "embedding-data/SPECTER": ["train"],
        # "embedding-data/PAQ_pairs": ["train"],
        # "embedding-data/WikiAnswers": ["train"],
    },
)

# Source: https://arxiv.org/pdf/1907.04307
use_multilingual_languages = [
    "ara-Arab",  # Arabic
    "zho-Hans",  # Chinese (Simplified, PRC)
    "zho-Hant",  # Chinese (Traditional, Taiwan)
    "nld-Latn",  # Dutch
    "eng-Latn",  # English
    "deu-Latn",  # German
    "fra-Latn",  # French
    "ita-Latn",  # Italian
    "por-Latn",  # Portuguese
    "spa-Latn",  # Spanish
    "jpn-Jpan",  # Japanese
    "kor-Kore",  # Korean
    "rus-Cyrl",  # Russian
    "pol-Latn",  # Polish
    "tha-Thai",  # Thai
    "tur-Latn",  # Turkish
]
use_multilingual_training_data = {
    # I'm not certain since they mined this themselves, but I would assume that there is significant overlap
    "StackOverflowQARetrieval": ["train", "test"],
    # Not in MTEB:
    # - SNLI translated to 15 languages (could have intersections with other NLI datasets)
    # - Translation pairs: Mined from the internet
    # - QA mined from Reddit, StackOverflow, YahooAnswers (could be problematic)
}
distiluse_base_multilingual_cased_v2 = ModelMeta(
    name="sentence-transformers/distiluse-base-multilingual-cased-v2",
    languages=use_multilingual_languages,
    open_weights=True,
    revision="dad0fa1ee4fa6e982d3adbce87c73c02e6aee838",
    release_date="2021-06-22",  # First commit
    n_parameters=135 * 1e6,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=True,
    public_training_data=True,
    training_datasets=use_multilingual_training_data,
)

use_cmlm_multilingual = ModelMeta(
    name="sentence-transformers/use-cmlm-multilingual",
    languages=paraphrase_langs,
    open_weights=True,
    revision="6f8ff6583c371cbc4d6d3b93a5e37a888fd54574",
    release_date="2022-04-14",  # First commit
    n_parameters=472 * 1e6,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=256,
    reference="https://huggingface.co/sentence-transformers/use-cmlm-multilingual",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from="sentence-transformers/LaBSE",
    public_training_code=True,
    public_training_data=True,
    training_datasets={
        # Not in MTEB:
        #  - SNLI
        #  - Translation corpus based largely on Uszkoreit et al. (2010)
    },
)


jina_embeddings_v2_base_en = ModelMeta(
    name="jinaai/jina-embeddings-v2-base-en",
    languages=["eng-Latn"],
    open_weights=True,
    revision="6e85f575bc273f1fd840a658067d0157933c83f0",
    release_date="2023-09-27",
    n_parameters=137_000_000,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/jinaai/jina-embeddings-v2-base-en",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets={"allenai/c4": ["train"]},
)

jina_embeddings_v2_small_en = ModelMeta(
    name="jinaai/jina-embeddings-v2-small-en",
    languages=["eng-Latn"],
    open_weights=True,
    revision="796cff318cdd4e5fbe8b7303a1ef8cbec36996ef",
    release_date="2023-09-27",
    n_parameters=32_700_000,
    memory_usage=None,
    embed_dim=512,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/jinaai/jina-embeddings-v2-small-en",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    training_datasets={"jinaai/negation-dataset": ["train"]},
)

jina_embedding_b_en_v1 = ModelMeta(
    name="jinaai/jina-embedding-b-en-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="aa0645035294a8c0607ce5bb700aba982cdff32c",
    release_date="2023-07-07",
    n_parameters=110_000_000,
    memory_usage=None,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/jinaai/jina-embedding-b-en-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by="jinaai/jina-embeddings-v2-base-en",
    adapted_from=None,
    training_datasets={"jinaai/negation-dataset": ["train"]},
)

jina_embedding_s_en_v1 = ModelMeta(
    name="jinaai/jina-embedding-s-en-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="c1fed70aa4823a640f1a7150a276e4d3b08dce08",
    release_date="2023-07-07",
    n_parameters=35_000_000,
    memory_usage=None,
    embed_dim=512,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/jinaai/jina-embedding-s-en-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by="jinaai/jina-embeddings-v2-small-en",
    adapted_from=None,
    training_datasets={"jinaai/negation-dataset": ["train"]},
)


all_MiniLM_L12_v2 = ModelMeta(
    name="sentence-transformers/all-MiniLM-L12-v2",
    languages=["eng-Latn"],
    open_weights=True,
    revision="364dd28d28dcd3359b537f3cf1f5348ba679da62",
    release_date="2021-08-30",
    n_parameters=33_400_000,
    memory_usage=None,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=256,
    reference="https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=False,  # does sentence transformer count?
    public_training_data=True,
    training_datasets={
        # source: frontmatter in readme
        # trained on stack exchange, unsure if sources match
        "StackExchangeClusteringP2P": ["test"],
        "StackExchangeClusteringP2P.v2": ["test"],
        "StackExchangeClustering": ["test"],
        "StackExchangeClustering.v2": ["test"],
        "NQ": ["test"],
        "NQHardNegatives": ["test"],
        "MSMARCO": ["train"],
        # Non MTEB sources
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
        # "trivia_qa": ["train"],
        # "embedding-data/sentence-compression": ["train"],
        # "embedding-data/flickr30k-captions": ["train"],
        # "embedding-data/altlex": ["train"],
        # "embedding-data/simple-wiki": ["train"],
        # "embedding-data/QQP": ["train"],
        # "embedding-data/SPECTER": ["train"],
        # "embedding-data/PAQ_pairs": ["train"],
        # "embedding-data/WikiAnswers": ["train"],
    },
)

microllama_text_embedding = ModelMeta(
    name="keeeeenw/MicroLlama-text-embedding",
    languages=["eng-Latn"],
    open_weights=True,
    revision="98f70f14cdf12d7ea217ed2fd4e808b0195f1e7e",
    release_date="2024-11-10",
    n_parameters=272_000_000,
    memory_usage=None,
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
        # shource yaml header:
        "NQ": ["test"]
        # not in MTEB:
        # "sentence-transformers/all-nli": ["train"],
        # "sentence-transformers/stsb": ["train"],
        # "sentence-transformers/quora-duplicates": ["train"],
        # "sentence-transformers/natural-questions": ["train"],
    },
)
