"""Bekko Static Embedding with the standard Sentence Transformers loader."""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

bekko_static_embedding_v1 = ModelMeta(
    name="hotchpotch/bekko-static-embedding-v1",
    revision="c18167b8f244c1bc6de21ce9b710bac8a29c4615",
    loader=SentenceTransformerEncoderWrapper,
    languages=[
        "eng-Latn",
        "arb-Arab",
        "bul-Cyrl",
        "ben-Beng",
        "cat-Latn",
        "ces-Latn",
        "dan-Latn",
        "deu-Latn",
        "ell-Grek",
        "spa-Latn",
        "ekk-Latn",
        "fas-Arab",
        "fin-Latn",
        "fra-Latn",
        "heb-Hebr",
        "hin-Deva",
        "hrv-Latn",
        "hun-Latn",
        "ind-Latn",
        "ita-Latn",
        "jpn-Jpan",
        "kor-Hang",
        "nld-Latn",
        "nob-Latn",
        "nno-Latn",
        "pol-Latn",
        "por-Latn",
        "ron-Latn",
        "rus-Cyrl",
        "slk-Latn",
        "slv-Latn",
        "srp-Cyrl",
        "srp-Latn",
        "swe-Latn",
        "swh-Latn",
        "tel-Telu",
        "tha-Thai",
        "tur-Latn",
        "ukr-Cyrl",
        "vie-Latn",
        "yor-Latn",
        "cmn-Hani",
    ],
    release_date="2026-10-21",
    n_parameters=204_800_000,
    n_embedding_parameters=204_800_000,
    memory_usage_mb=391,
    max_tokens=32768,
    embed_dim=[32, 64, 128, 256, 512, 1024],
    license="mit",
    open_weights=True,
    public_training_code=(
        "https://huggingface.co/hotchpotch/bekko-static-embedding-v1/"
        "tree/c18167b8f244c1bc6de21ce9b710bac8a29c4615/training_scripts"
    ),
    public_training_data=(
        "https://huggingface.co/datasets/hotchpotch/bekko-embedding-v1-unsupervised"
    ),
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    reference="https://huggingface.co/hotchpotch/bekko-static-embedding-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    # Corpus-level exposure from the full 265-config training selection:
    # https://huggingface.co/hotchpotch/bekko-static-embedding-v1/blob/c18167b8f244c1bc6de21ce9b710bac8a29c4615/training_scripts/bekko-unsupervised-multilingual-40-direct.json
    # The public_training_data card documents every upstream source and the
    # individual Nomic/LightOn subsets. Use MTEB task names here; sources without
    # MTEB counterparts remain documented in that card. These conservative
    # corpus-family disclosures do not assert evaluation-row duplication.
    training_datasets={
        # sentence-transformers/miracl (train split).
        "MIRACLRetrieval",
        "MIRACLRetrievalHardNegatives",
        # sentence-transformers/all-nli (SNLI/MultiNLI) and their translations.
        "XNLI",
        "XNLIV2",
        "AfriXNLI",
        "IndicXnliPairClassification",
        "Cmnli",
        "KorNLI",
        # LightOn: beir_dbpedia and webfaq_eng.
        "DBPedia",
        "DBpediaClassification",
        "WebFAQRetrieval",
        "WebFAQBitextMiningQuestions",
        "WebFAQBitextMiningQAs",
        # Nomic: amazon_reviews/amazonqa.
        "AmazonPolarityClassification",
        "AmazonReviewsClassification",
        "AmazonCounterfactualClassification",
        # Nomic: codesearch.
        "CodeSearchNetRetrieval",
        "CodeSearchNetCCRetrieval",
        "COIRCodeSearchNetRetrieval",
        # Nomic: yahoo_qa/yahoo_title_answer/yahoo_title_question.
        "YahooAnswersTopicsClassification",
        # Nomic: quora.
        "QuoraRetrieval",
        # Nomic: reddit_title_body.
        "RedditClustering",
        "RedditClusteringP2P",
        # Nomic: stackexchange_body_body/duplicate_questions/title_body.
        "StackExchangeClustering",
        "StackExchangeClusteringP2P",
        "AskUbuntuDupQuestions",
        "StackOverflowDupQuestions",
        "CQADupstackRetrieval",
        "StackOverflowQA",
        # Nomic: s2orc_abstract_citation/citation_titles/title_abstract.
        "SCIDOCS",
        "SciDocsRR",
        # hotchpotch/arxiv-ir-simulated-search-queries and S2ORC.
        "ArxivClassification",
        "ArxivClusteringP2P",
        "ArxivClusteringS2S",
        "ArXivHierarchicalClusteringP2P",
        "ArXivHierarchicalClusteringS2S",
        # Wikipedia/FineWiki IR, related-document, synthetic-NLI and SWIM sources.
        "WikipediaRetrievalMultilingual",
        "WikipediaRerankingMultilingual",
        "WikiCitiesClustering",
    },
    citation="""@misc{tateno2026bekko,
  title={Bekko Embedding: Parameter-Efficient Multilingual Retrieval with Ultra-Compact Encoders},
  author={Yuichi Tateno},
  year={2026},
  eprint={2607.25180},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2607.25180},
}""",
    adapted_from=None,
    modalities=["text"],
    model_type=["dense"],
    contacts=["hotchpotch"],
)
