"""Models for GeoGPT-Research-Project"""

from __future__ import annotations

import torch

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta

geoembedding = ModelMeta.model_construct(
    name="GeoGPT-Research-Project/GeoEmbedding",
    languages=["eng-Latn"],
    open_weights=True,
    revision="29803c28ea7ef6871194a8ebc85ad7bfe174928e",
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template="Instruct: {instruction}\nQuery: ",
        apply_instruction_to_passages=False,
        model_kwargs={"torch_dtype": torch.bfloat16},
        trust_remote_code=True,
    ),
    release_date="2025-04-22",
    n_parameters=7241732096,
    memory_usage_mb=27625,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/GeoGPT-Research-Project/GeoEmbedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets={
        "ArguAna": ["test"],
        "FEVER": ["train"],
        "MSMARCO": ["train"],
        "FiQA2018": ["train"],
        "HotpotQA": ["train"],
        "NFCorpus": ["train"],
        "SciFact": ["train"],
        "AmazonCounterfactualClassification": ["train"],
        "AmazonPolarityClassification": ["train"],
        "AmazonReviewsClassification": ["train"],
        "Banking77Classification": ["train"],
        "EmotionClassification": ["train"],
        "MassiveIntentClassification": ["train"],
        "MTOPDomainClassification": ["train"],
        "MTOPIntentClassification": ["train"],
        "ToxicConversationsClassification": ["train"],
        "TweetSentimentExtractionClassification": ["train"],
        "ArxivClusteringS2S": ["test"],
        "ArxivClusteringP2P": ["test"],
        "MedrixvClusteringS2S": ["test"],
        "MedrixvClusteringP2P": ["test"],
        "BiorxivClusteringS2S": ["test"],
        "BiorxivClusteringP2P": ["test"],
        "TwentyNewsgroupsClustering": ["test"],
        "STS12": ["train"],
        "STS22": ["train"],
        "STSBenchmark": ["train"],
        "StackOverflowDupQuestions": ["train"],
    },
)
