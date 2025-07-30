from __future__ import annotations

from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

from .e5_instruct import E5_MISTRAL_TRAINING_DATA

LGAI_EMBEDDING_TRAINING_DATA = {
    # source: https://arxiv.org/abs/2506.07438
    "ArguAna",
    "ELI5",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NQ",
    "QuoraDuplicateQuestions",
    "SciDocsReranking",
    "SQuAD",
    "StackOverflowDupQuestions",
    "AmazonCounterfactualClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    "ArxivClusteringS2S",
    "ArxivClusteringP2P",
    "BiorxivClusteringS2S",
    "BiorxivClusteringP2P",
    "MedrxivClusteringS2S",
    "MedrxivClusteringP2P",
    "RedditClusteringP2P",
    "RedditClustering",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
    "STS12",
    "STS22",
    "STSBenchmark",
    "SNLI",
}

lgai_embedding_en = ModelMeta(
    loader=sentence_transformers_loader,
    name="annamodels/LGAI-Embedding-Preview",
    languages=[
        "eng-Latn",
    ],
    open_weights=True,
    revision="5e0b2316acc8c2e2941ded6b9cb200b1cb313e65",
    release_date="2025-06-11",
    n_parameters=7_110_000_000,
    memory_usage_mb=27125,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/annamodels/LGAI-Embedding-Preview",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    adapted_from="mistralai/Mistral-7B-v0.1",
    training_datasets=E5_MISTRAL_TRAINING_DATA | LGAI_EMBEDDING_TRAINING_DATA,
)
