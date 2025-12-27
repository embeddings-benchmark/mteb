import torch

from mteb.models.instruct_wrapper import (
    InstructSentenceTransformerModel,
    instruct_wrapper,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction

from .e5_models import (
    E5_PAPER_RELEASE_DATE,
    ME5_TRAINING_DATA,
    XLMR_LANGUAGES,
)

MISTRAL_LANGUAGES = ["eng-Latn", "fra-Latn", "deu-Latn", "ita-Latn", "spa-Latn"]

E5_INSTRUCTION = "Instruct: {instruction}\nQuery: "

E5_MISTRAL_TRAINING_DATA = {
    "FEVER",
    "FEVERHardNegatives",
    "FEVER-NL",  # translation not trained on
    "HotpotQA",
    "HotpotQAHardNegatives",
    "HotpotQA-PL",  # translation not trained on
    "HotpotQA-NL",  # translation not trained on
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MIRACLReranking",  # https://arxiv.org/pdf/2402.09906, section M
} | ME5_TRAINING_DATA

e5_instruct = ModelMeta(
    loader=instruct_wrapper,
    loader_kwargs=dict(
        instruction_template=E5_INSTRUCTION,
        attn="cccc",
        pooling_method="mean",
        mode="embedding",
        torch_dtype=torch.float16,
        normalized=True,
    ),
    name="intfloat/multilingual-e5-large-instruct",
    model_type=["dense"],
    languages=XLMR_LANGUAGES,
    open_weights=True,
    revision="baa7be480a7de1539afce709c8f13f833a510e0a",
    release_date=E5_PAPER_RELEASE_DATE,
    framework=["GritLM", "PyTorch", "Sentence Transformers"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    reference="https://huggingface.co/intfloat/multilingual-e5-large-instruct",
    n_parameters=560_000_000,
    memory_usage_mb=1068,
    embed_dim=1024,
    license="mit",
    max_tokens=514,
    adapted_from="FacebookAI/xlm-roberta-large",
    citation="""@article{wang2024multilingual,
      title={Multilingual E5 Text Embeddings: A Technical Report},
      author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
      journal={arXiv preprint arXiv:2402.05672},
      year={2024}
    }""",
    public_training_code=None,
    public_training_data=None,
    training_datasets=ME5_TRAINING_DATA,
)

e5_mistral = ModelMeta(
    loader=instruct_wrapper,
    loader_kwargs=dict(
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
    model_type=["dense"],
    languages=MISTRAL_LANGUAGES,
    open_weights=True,
    revision="07163b72af1488142a360786df853f237b1a3ca1",
    release_date=E5_PAPER_RELEASE_DATE,
    framework=["GritLM", "PyTorch", "Sentence Transformers"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    reference="https://huggingface.co/intfloat/e5-mistral-7b-instruct",
    n_parameters=7_111_000_000,
    memory_usage_mb=13563,
    embed_dim=4096,
    license="mit",
    max_tokens=32768,
    citation="""
    @article{wang2023improving,
      title={Improving Text Embeddings with Large Language Models},
      author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
      journal={arXiv preprint arXiv:2401.00368},
      year={2023}
    }

    @article{wang2022text,
      title={Text Embeddings by Weakly-Supervised Contrastive Pre-training},
      author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Jiao, Binxing and Yang, Linjun and Jiang, Daxin and Majumder, Rangan and Wei, Furu},
      journal={arXiv preprint arXiv:2212.03533},
      year={2022}
    }
    """,
    public_training_code=None,
    public_training_data=None,
    training_datasets=E5_MISTRAL_TRAINING_DATA,
    adapted_from="mistralai/Mistral-7B-v0.1",
)

zeta_alpha_ai__zeta_alpha_e5_mistral = ModelMeta(
    loader=instruct_wrapper,
    loader_kwargs=dict(
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
    model_type=["dense"],
    revision="c791d37474fa6a5c72eb3a2522be346bc21fbfc3",
    release_date="2024-08-30",
    languages=["eng-Latn"],
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
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets={
        # copied from e5
        # source: https://arxiv.org/pdf/2212.03533
        "NQ",
        "NQ-NL",  # translation not trained on
        "NQHardNegatives",
        "MSMARCO",  # dev?
        "mMARCO-NL",  # translation not trained on
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
        # "ArxivClusteringS2S",
        # "ArxivClusteringP2P",
        # https://huggingface.co/datasets/mteb/raw_biorxiv
        # "BiorxivClusteringS2S",
        # "BiorxivClusteringP2P",
        # https://huggingface.co/datasets/mteb/raw_medrxiv
        # "MedrxivClusteringS2S",
        # "MedrxivClusteringP2P",
        # as their train datasets
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "Banking77Classification",
        "EmotionClassification",
        "MTOPIntentClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
        "ImdbClassification",
        "STS12",
        "STS22",
        "STSBenchmark",
        "MIRACLRetrieval",
        "MIRACLRetrievalHardNegatives",
        "MIRACLReranking",  # https://arxiv.org/pdf/2402.05672, table 2
    }
    | E5_MISTRAL_TRAINING_DATA,
    adapted_from="intfloat/e5-mistral-7b-instruct",
    superseded_by=None,
)

E5_R_MISTRAL_7B_INSTRUCTION = "{instruction}\n"
BeastyZ__e5_R_mistral_7b = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=E5_R_MISTRAL_7B_INSTRUCTION,
        tokenizer_kwargs={"pad_token": "</s>"},
    ),
    name="BeastyZ/e5-R-mistral-7b",
    model_type=["dense"],
    revision="3f810a6a7fd220369ad248e3705cf13d71803602",
    release_date="2024-06-28",
    languages=["eng-Latn"],
    n_parameters=7241732096,
    memory_usage_mb=27625,
    max_tokens=32768.0,
    embed_dim=4096,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/LeeSureman/E5-Retrieval-Reproduction",
    public_training_data="https://huggingface.co/datasets/BeastyZ/E5-R",
    framework=["PyTorch"],
    reference="https://huggingface.co/BeastyZ/e5-R-mistral-7b",
    similarity_fn_name="cosine",
    use_instructions=True,
    training_datasets=E5_MISTRAL_TRAINING_DATA,
    # not MTEB: {"BeastyZ/E5-R"},
    adapted_from="intfloat/e5-mistral-7b-instruct",
    superseded_by=None,
)
