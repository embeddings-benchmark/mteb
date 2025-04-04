from __future__ import annotations

from functools import partial

import torch

from mteb.model_meta import ModelMeta
from mteb.models.e5_models import (
    E5_PAPER_RELEASE_DATE,
    ME5_TRAINING_DATA,
    XLMR_LANGUAGES,
)
from mteb.models.instruct_wrapper import instruct_wrapper

MISTRAL_LANGUAGES = ["eng_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "spa_Latn"]

E5_INSTRUCTION = "Instruct: {instruction}\nQuery: "

E5_MISTRAL_TRAINING_DATA = {
    **ME5_TRAINING_DATA,
    "FEVER": ["train"],
    "FEVERHardNegatives": ["train"],
    "FEVER-NL": ["train"],  # translation not trained on
    "HotpotQA": ["train"],
    "HotpotQAHardNegatives": ["train"],
    "HotpotQA-PL": ["train"],  # translation not trained on
    "HotpotQA-NL": ["train"],  # translation not trained on
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
    "MIRACLReranking": ["train"],  # https://arxiv.org/pdf/2402.09906, section M
}

e5_instruct = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="intfloat/multilingual-e5-large-instruct",
        instruction_template=E5_INSTRUCTION,
        attn="cccc",
        pooling_method="mean",
        mode="embedding",
        torch_dtype=torch.float16,
        normalized=True,
    ),
    name="intfloat/multilingual-e5-large-instruct",
    languages=XLMR_LANGUAGES,
    open_weights=True,
    revision="baa7be480a7de1539afce709c8f13f833a510e0a",
    release_date=E5_PAPER_RELEASE_DATE,
    framework=["GritLM", "PyTorch", "Sentence Transformers"],
    similarity_fn_name="cosine",
    use_instructions=True,
    reference="https://huggingface.co/intfloat/multilingual-e5-large-instruct",
    n_parameters=560_000_000,
    memory_usage_mb=1068,
    embed_dim=1024,
    license="mit",
    max_tokens=514,
    adapted_from="FacebookAI/xlm-roberta-large",
    public_training_code=None,
    public_training_data=None,
    training_datasets=ME5_TRAINING_DATA,
)

e5_mistral = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="intfloat/e5-mistral-7b-instruct",
        instruction_template=E5_INSTRUCTION,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.float16,
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/intfloat/e5-mistral-7b-instruct#transformers
        normalized=True,
    ),
    name="intfloat/e5-mistral-7b-instruct",
    languages=MISTRAL_LANGUAGES,
    open_weights=True,
    revision="07163b72af1488142a360786df853f237b1a3ca1",
    release_date=E5_PAPER_RELEASE_DATE,
    framework=["GritLM", "PyTorch", "Sentence Transformers"],
    similarity_fn_name="cosine",
    use_instructions=True,
    reference="https://huggingface.co/intfloat/e5-mistral-7b-instruct",
    n_parameters=7_111_000_000,
    memory_usage_mb=13563,
    embed_dim=4096,
    license="mit",
    max_tokens=32768,
    public_training_code=None,
    public_training_data=None,
    training_datasets=E5_MISTRAL_TRAINING_DATA,
    adapted_from="mistralai/Mistral-7B-v0.1",
)

zeta_alpha_ai__Zeta_Alpha_E5_Mistral = ModelMeta(
    loader=partial(  # type: ignore
        instruct_wrapper,
        model_name_or_path="zeta-alpha-ai/Zeta-Alpha-E5-Mistral",
        instruction_template=E5_INSTRUCTION,
        attn="cccc",
        pooling_method="lasttoken",
        mode="embedding",
        torch_dtype=torch.bfloat16,
        # The ST script does not normalize while the HF one does so unclear what to do
        # https://huggingface.co/intfloat/e5-mistral-7b-instruct#transformers
        normalized=True,
    ),
    name="zeta-alpha-ai/Zeta-Alpha-E5-Mistral",
    revision="c791d37474fa6a5c72eb3a2522be346bc21fbfc3",
    release_date="2024-08-30",
    languages=["eng_Latn"],
    n_parameters=7110660096,
    memory_usage_mb=13563,
    max_tokens=32768.0,
    embed_dim=4096,
    license="mit",
    open_weights=True,
    public_training_data=None,
    public_training_code=None,
    framework=["PyTorch", "Sentence Transformers", "GritLM"],
    reference="https://huggingface.co/zeta-alpha-ai/Zeta-Alpha-E5-Mistral",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets={
        **E5_MISTRAL_TRAINING_DATA,
        # copied from e5
        # source: https://arxiv.org/pdf/2212.03533
        "NQ": ["test"],
        "NQ-NL": ["test"],  # translation not trained on
        "NQHardNegatives": ["test"],
        "MSMARCO": ["train"],  # dev?
        "mMARCO-NL": ["train"],  # translation not trained on
        # source: https://www.zeta-alpha.com/post/fine-tuning-an-llm-for-state-of-the-art-retrieval-zeta-alpha-s-top-10-submission-to-the-the-mteb-be
        # "Arguana",
        # "FEVER",
        # "FIQA",
        # "HotPotQA",
        # "MsMarco (passage)",
        # "NFCorpus",
        # "SciFact",
        # "NLI",
        # "SQuad",
        # "StackExchange",
        # "TriviaQA",
        # "SciRep",
        # "SciRepEval"
        # mteb
        # https://huggingface.co/datasets/mteb/raw_arxiv
        # "ArxivClusteringS2S": ["train"],
        # "ArxivClusteringP2P": ["train"],
        # https://huggingface.co/datasets/mteb/raw_biorxiv
        # "BiorxivClusteringS2S": ["train"],
        # "BiorxivClusteringP2P": ["train"],
        # https://huggingface.co/datasets/mteb/raw_medrxiv
        # "MedrxivClusteringS2S": ["train"],
        # "MedrxivClusteringP2P": ["train"],
        # as their train datasets
        "AmazonCounterfactualClassification": ["train"],
        "AmazonReviewsClassification": ["train"],
        "Banking77Classification": ["train"],
        "EmotionClassification": ["train"],
        "MTOPIntentClassification": ["train"],
        "ToxicConversationsClassification": ["train"],
        "TweetSentimentExtractionClassification": ["train"],
        "ImdbClassification": ["train"],
        "STS12": ["train"],
        "STS22": ["train"],
        "STSBenchmark": ["train"],
        "MIRACLRetrieval": ["train"],
        "MIRACLRetrievalHardNegatives": ["train"],
        "MIRACLReranking": ["train"],  # https://arxiv.org/pdf/2402.05672, table 2
    },
    adapted_from="intfloat/e5-mistral-7b-instruct",
    superseded_by=None,
)
