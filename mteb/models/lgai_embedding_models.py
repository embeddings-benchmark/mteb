from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader
from mteb.models.e5_instruct import E5_MISTRAL_TRAINING_DATA

LGAI_EMBEDDING_TRAINING_DATA = {
    # source: https://arxiv.org/abs/2506.07438
    "ArguAna": ["train"],
    "ELI5": ["train"],
    "FEVER": ["train"],
    "FiQA2018": ["train"],
    "HotpotQA": ["train"],
    "MSMARCO": ["train"],
    "NQ": ["train"],
    "QuoraDuplicateQuestions": ["train"],
    "SciDocsReranking": ["train"],
    "SQuAD": ["train"],
    "StackOverflowDupQuestions": ["train"],
    "AmazonCounterfactualClassification": ["train"],
    "AmazonReviewsClassification": ["train"],
    "Banking77Classification": ["train"],
    "EmotionClassification": ["train"],
    "ImdbClassification": ["train"],
    "MTOPIntentClassification": ["train"],
    "ToxicConversationsClassification": ["train"],
    "TweetSentimentExtractionClassification": ["train"],
    "ArxivClusteringS2S": ["train"],
    "ArxivClusteringP2P": ["train"],
    "BiorxivClusteringS2S": ["train"],
    "BiorxivClusteringP2P": ["train"],
    "MedrxivClusteringS2S": ["train"],
    "MedrxivClusteringP2P": ["train"],
    "RedditClusteringP2P": ["train"],
    "RedditClustering": ["train"],
    "StackExchangeClustering": ["train"],
    "StackExchangeClusteringP2P": ["train"],
    "TwentyNewsgroupsClustering": ["train"],
    "STS12": ["train"],
    "STS22": ["train"],
    "STSBenchmark": ["train"],
    "SNLI": ["train"],
}

lgai_embedding_en = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="annamodels/LGAI-Embedding-Preview",
        revision="5e0b2316acc8c2e2941ded6b9cb200b1cb313e65",
    ),
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
    training_datasets={
        **E5_MISTRAL_TRAINING_DATA,
        **LGAI_EMBEDDING_TRAINING_DATA,
    },
)
