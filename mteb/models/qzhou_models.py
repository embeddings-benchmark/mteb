from __future__ import annotations

import os
from functools import partial
from mteb.model_meta import ModelMeta
from mteb.encoder_interface import PromptType
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper
from mteb.models.e5_instruct import E5_MISTRAL_TRAINING_DATA

def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.passage:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"



def qzhou_instruct_loader(model_name, **kwargs):
    model = InstructSentenceTransformerWrapper(
        model_name,
        revision=kwargs.pop("model_revision", None),
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        **kwargs,
    )
    encoder = model.model._first_module()
    encoder.tokenizer.padding_side = "left"
    return model

bge_m3_training_data_qzhou_use = {
    # source: https://arxiv.org/abs/2402.03216
    "MIRACLRetrieval": ["train"],
    "CMedQAv1-reranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "MrTidyRetrieval": ["train"],
    "T2Reranking": ["train"],
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NQ": ["train"],
    "HotpotQA": ["train"],
    "HotpotQAHardNegatives": ["train"],
    "T2Retrieval": ["train"],
    "DuRetrieval": ["train"],
    "MMarcoReranking": ["train"]
}

bge_chinese_training_data_qzhou_use = {
    # source: https://arxiv.org/pdf/2309.07597
    "T2Retrieval": ["train"],
    "DuRetrieval": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "Cmnli": ["train"],
    "Ocnli": ["train"],
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NQ": ["test"],
    "NQHardNegatives": ["test"],
    "HotpotQA": ["train"],
    "HotpotQAHardNegatives": ["train"],
    "QuoraRetrieval": ["train"],
    "QuoraRetrievalHardNegatives": ["train"],
}


bge_full_data_qzhou_use = {
    # source: https://arxiv.org/pdf/2409.15700
    "HotpotQA": ["train"],
    "FEVER": ["train"],
    "MSMARCO": ["train"],
    "NQ": ["train"],
    "ArguAna": ["train"],
    "FiQA2018": ["train"],
    "SciDocsReranking": ["train"],
    "StackOverflowDupQuestions": ["train"],
    "AmazonReviewsClassification": ["train"],
    "AmazonCounterfactualClassification": ["train"],
    "Banking77Classification": ["train"],
    "EmotionClassification": ["train"],
    "TweetSentimentExtractionClassification": ["train"],
    "MTOPIntentClassification": ["train"],
    "ImdbClassification": ["train"],
    "ToxicConversationsClassification": ["train"],
    "ArxivClusteringS2S": ["train"],
    "ArxivClusteringP2P": ["train"],
    "BiorxivClusteringS2S": ["train"],
    "BiorxivClusteringP2P": ["train"],
    "MedrxivClusteringS2S": ["train"],
    "MedrxivClusteringP2P": ["train"],
    "BiorxivClusteringS2S.v2": ["train"],
    "BiorxivClusteringP2P.v2": ["train"],
    "MedrxivClusteringS2S.v2": ["train"],
    "MedrxivClusteringP2P.v2": ["train"],
    "RedditClusteringP2P": ["train"],
    "RedditClustering": ["train"],
    "RedditClustering.v2": ["train"],
    "TwentyNewsgroupsClustering": ["train"],
    "TwentyNewsgroupsClustering.v2": ["train"],
    "STS22": ["train"],
    "STS22.v2": ["train"],
    "STSBenchmark": ["train"],
}


QZhou_Embedding = ModelMeta(
    loader = partial(
        qzhou_instruct_loader,
        model_name="Kingsoft-LLM/QZhou-Embedding",
        model_revision="b43142d518d6e5251fd2d1e0a8741eef5c8b980a",
    ),
    name="Kingsoft-LLM/QZhou-Embedding",
    languages=["eng-Latn", "zho-Hans"], 
    open_weights=True,
    revision="b43142d518d6e5251fd2d1e0a8741eef5c8b980a",
    release_date="2025-08-01",
    n_parameters=7_070_619_136,
    memory_usage_mb=29070,
    embed_dim=3584,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/Kingsoft-LLM/QZhou-Embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/cfli/datasets",
    training_datasets={
        **bge_m3_training_data_qzhou_use,
        **bge_chinese_training_data_qzhou_use,
        **bge_full_data_qzhou_use,
        "Shitao/MLDR": ["train"],
        "FreedomIntelligence/Huatuo26M-Lite": ["train"],
        "infgrad/retrieval_data_llm": ["train"],
    },
)
