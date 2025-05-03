from __future__ import annotations

import logging
from functools import partial

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper

logger = logging.getLogger(__name__)


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return f"Instruct: {instruction}\nQuery: " if instruction else ""


nvidia_training_datasets = {
    # source: https://arxiv.org/pdf/2405.17428
    "ArguAna": ["train"],
    "ArguAna-PL": ["train"],
    "ArguAna-NL": ["train"],  # translation not trained on
    "NanoArguAnaRetrieval": ["train"],
    "HotpotQA": ["train"],
    "HotpotQA-PL": ["train"],  # translation not trained on
    "HotpotQA-NL": ["train"],  # translation not trained on
    "HotpotQAHardNegatives": ["train"],
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
    "FEVER": ["train"],
    "FEVER-NL": ["train"],  # translation not trained on
    "FEVERHardNegatives": ["train"],
    "NanoFEVERRetrieval": ["train"],
    "FiQA2018": ["train"],
    "FiQA2018-NL": ["train"],  # translation not trained on
    "STS12": ["train"],
    "STS22": ["train"],
    "AmazonReviewsClassification": ["train"],
    "AmazonCounterfactualClassification": ["train"],
    "Banking77Classification": ["train"],
    "EmotionClassification": ["train"],
    "ImdbClassification": ["train"],
    "MTOPIntentClassification": ["train"],
    "ToxicConversationsClassification": ["train"],
    "TweetSentimentExtractionClassification": ["train"],
    "ArxivClusteringP2P": ["train"],
    "ArxivClusteringP2P.v2": ["train"],
    "ArxivClusteringS2S": ["train"],
    "BiorxivClusteringP2P": ["train"],
    "BiorxivClusteringP2P.v2": ["train"],
    "BiorxivClusteringS2S": ["train"],
    "BiorxivClusteringS2S.v2": ["train"],
    "MedrxivClusteringP2P": ["train"],
    "MedrxivClusteringP2P.v2": ["train"],
    "MedrxivClusteringS2S": ["train"],
    "MedrxivClusteringS2S.v2": ["train"],
    "TwentyNewsgroupsClustering": ["train"],
    "TwentyNewsgroupsClustering.v2": ["train"],
    "StackExchangeClustering": ["train"],
    "StackExchangeClustering.v2": ["train"],
    "StackExchangeClusteringP2P": ["train"],
    "StackExchangeClusteringP2P.v2": ["train"],
    "RedditClustering": ["train"],
    "RedditClustering.v2": ["train"],
    "RedditClusteringP2P": ["train"],
    "RedditClusteringP2P.v2": ["train"],
    "STSBenchmark": ["train"],
    "STSBenchmarkMultilingualSTS": ["train"],  # translated, not trained on
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
    "MIRACLReranking": ["train"],
    "MrTidyRetrieval": ["train"],
}

NV_embed_v2 = ModelMeta(
    loader=partial(  # type: ignore
        InstructSentenceTransformerWrapper,
        model_name="nvidia/NV-Embed-v2",
        revision="7604d305b621f14095a1aa23d351674c2859553a",
        instruction_template=instruction_template,
        trust_remote_code=True,
        max_seq_length=32768,
        padding_side="right",
        # for nv-embed, we add eos token to each input example
        add_eos_token=True,
    ),
    name="nvidia/NV-Embed-v2",
    languages=["eng-Latn"],
    open_weights=True,
    revision="7604d305b621f14095a1aa23d351674c2859553a",
    release_date="2024-09-09",  # initial commit of hf model.
    n_parameters=7_850_000_000,
    memory_usage_mb=14975,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/nvidia/NV-Embed-v2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=nvidia_training_datasets,
    public_training_code=None,
    public_training_data=None,
)

NV_embed_v1 = ModelMeta(
    loader=partial(  # type: ignore
        InstructSentenceTransformerWrapper,
        model_name="nvidia/NV-Embed-v1",
        revision="7604d305b621f14095a1aa23d351674c2859553a",
        instruction_template=instruction_template,
        trust_remote_code=True,
        max_seq_length=32768,
        padding_side="right",
        # for nv-embed, we add eos token to each input example
        add_eos_token=True,
    ),
    name="nvidia/NV-Embed-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="570834afd5fef5bf3a3c2311a2b6e0a66f6f4f2c",
    release_date="2024-09-13",  # initial commit of hf model.
    n_parameters=7_850_000_000,
    memory_usage_mb=29945,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/nvidia/NV-Embed-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=nvidia_training_datasets,
    public_training_code=None,
    public_training_data=None,
)
