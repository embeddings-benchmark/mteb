from __future__ import annotations

from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import CrossEncoderWrapper

_ETTIN_V1_CITATION = """@misc{aarsen2026ettin-reranker,
    title = "Introducing the Ettin Reranker Family",
    author = "Aarsen, Tom",
    year = "2026",
    publisher = "Hugging Face",
    url = "https://huggingface.co/blog/ettin-reranker",
}
"""

_ETTIN_V1_TRAINING_DATA = {
    # source: https://huggingface.co/datasets/cross-encoder/ettin-reranker-v1-data
    # rerank-scored retrieval data
    "FEVER",
    "FEVERHardNegatives.v2",
    "NanoFEVERRetrieval",
    "FiQA2018",
    "NanoFiQA2018Retrieval",
    "HotpotQA",
    "HotpotQAHardNegatives.v2",
    "NanoHotpotQARetrieval",
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
    "NQ",
    "NQHardNegatives",
    "NanoNQRetrieval",
    # not in MTEB: SQuADv2, TriviaQA
    # LightOn pre-training data
    "ArXivHierarchicalClusteringP2P",
    "ArXivHierarchicalClusteringS2S",
    "AmazonReviewsClassification",
    "AmazonPolarityClassification.v2",
    "AmazonCounterfactualClassification",
    "BiorxivClusteringP2P.v2",
    "BiorxivClusteringS2S.v2",
    "DBPedia",
    "DBPediaHardNegatives.v2",
    "NanoDBPediaRetrieval",
    "MedrxivClusteringP2P.v2",
    "MedrxivClusteringS2S.v2",
    "QuoraRetrieval",
    "QuoraRetrievalHardNegatives.v2",
    "NanoQuoraRetrieval",
    "RedditClustering.v2",
    "RedditClusteringP2P.v2",
    "SCIDOCS",
    "SciDocsRR",
    "NanoSCIDOCSRetrieval",
    "StackExchangeClustering.v2",
    "StackExchangeClusteringP2P.v2",
    "CQADupstackRetrieval",
    "AskUbuntuDupQuestions",
    "SummEvalSummarization.v2",
    "YahooAnswersTopicsClassification.v2",
    # not in MTEB: agnews, altlex, amazon_qa, cc_news_en, fw_edu, gooaq_qa,
    #              mtp, npr, paq, wikianswers, wikihow, SQuADv2, TriviaQA
}

_ETTIN_V1_COMMON = dict(
    loader=CrossEncoderWrapper,
    model_type=["cross-encoder"],
    languages=["eng-Latn"],
    open_weights=True,
    release_date="2026-05-15",
    max_tokens=7999,
    embed_dim=None,
    license="apache-2.0",
    public_training_code="https://huggingface.co/blog/ettin-reranker#overall-training-script",
    public_training_data="https://huggingface.co/datasets/cross-encoder/ettin-reranker-v1-data",
    similarity_fn_name=None,
    use_instructions=None,
    training_datasets=_ETTIN_V1_TRAINING_DATA,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    citation=_ETTIN_V1_CITATION,
    contacts=["tomaarsen"],
)

ettin_reranker_17m_v1 = ModelMeta(
    **_ETTIN_V1_COMMON,
    name="cross-encoder/ettin-reranker-17m-v1",
    revision="9e4aa35321a6dd1a43ca313f500c4b4f7cfb5cc6",
    n_parameters=16_797_440,
    n_embedding_parameters=12_894_208,
    memory_usage_mb=64,
    adapted_from="https://huggingface.co/jhu-clsp/ettin-encoder-17m",
    reference="https://huggingface.co/cross-encoder/ettin-reranker-17m-v1",
)

ettin_reranker_32m_v1 = ModelMeta(
    **_ETTIN_V1_COMMON,
    name="cross-encoder/ettin-reranker-32m-v1",
    revision="b33e5ceb5110773ea9cf5e00c9bedc83a8c2afdd",
    n_parameters=31_883_136,
    n_embedding_parameters=19_341_312,
    memory_usage_mb=122,
    adapted_from="https://huggingface.co/jhu-clsp/ettin-encoder-32m",
    reference="https://huggingface.co/cross-encoder/ettin-reranker-32m-v1",
)

ettin_reranker_68m_v1 = ModelMeta(
    **_ETTIN_V1_COMMON,
    name="cross-encoder/ettin-reranker-68m-v1",
    revision="d166fa88ddde3c42bc3ee92f7df476d941c8204a",
    n_parameters=68_144_640,
    n_embedding_parameters=25_788_416,
    memory_usage_mb=260,
    adapted_from="https://huggingface.co/jhu-clsp/ettin-encoder-68m",
    reference="https://huggingface.co/cross-encoder/ettin-reranker-68m-v1",
)

ettin_reranker_150m_v1 = ModelMeta(
    **_ETTIN_V1_COMMON,
    name="cross-encoder/ettin-reranker-150m-v1",
    revision="025501c4e0f9bbeb4c5b198318e0089ff061cc14",
    n_parameters=149_014_272,
    n_embedding_parameters=38_682_624,
    memory_usage_mb=568,
    adapted_from="https://huggingface.co/jhu-clsp/ettin-encoder-150m",
    reference="https://huggingface.co/cross-encoder/ettin-reranker-150m-v1",
)

ettin_reranker_400m_v1 = ModelMeta(
    **_ETTIN_V1_COMMON,
    name="cross-encoder/ettin-reranker-400m-v1",
    revision="5dca36282a5d85f368d2544002513a29159b4c9e",
    n_parameters=394_781_696,
    n_embedding_parameters=51_576_832,
    memory_usage_mb=1506,
    adapted_from="https://huggingface.co/jhu-clsp/ettin-encoder-400m",
    reference="https://huggingface.co/cross-encoder/ettin-reranker-400m-v1",
)

ettin_reranker_1b_v1 = ModelMeta(
    **_ETTIN_V1_COMMON,
    name="cross-encoder/ettin-reranker-1b-v1",
    revision="7d20e9baad17016fdf5549c08f69a2d7ca3e60c3",
    n_parameters=1_028_050_688,
    n_embedding_parameters=90_259_456,
    memory_usage_mb=3922,
    adapted_from="https://huggingface.co/jhu-clsp/ettin-encoder-1b",
    reference="https://huggingface.co/cross-encoder/ettin-reranker-1b-v1",
)
