import logging

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

logger = logging.getLogger(__name__)


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return f"Instruct: {instruction}\nQuery: " if instruction else ""


nvidia_training_datasets = {
    # source: https://arxiv.org/pdf/2405.17428
    "ArguAna",
    "ArguAna-PL",
    "ArguAna-NL",  # translation not trained on
    "NanoArguAnaRetrieval",
    "HotpotQA",
    "HotpotQA-PL",  # translation not trained on
    "HotpotQA-NL",  # translation not trained on
    "HotpotQAHardNegatives",
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
    "MSMARCO-PL",  # translation not trained on
    "mMARCO-NL",  # translation not trained on
    "NQ",
    "NQHardNegatives",
    "NanoNQRetrieval",
    "NQ-PL",  # translation not trained on
    "NQ-NL",  # translation not trained on
    "FEVER",
    "FEVER-NL",  # translation not trained on
    "FEVERHardNegatives",
    "NanoFEVERRetrieval",
    "FiQA2018",
    "FiQA2018-NL",  # translation not trained on
    "STS12",
    "STS22",
    "AmazonReviewsClassification",
    "AmazonCounterfactualClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    "ArxivClusteringP2P",
    "ArxivClusteringP2P.v2",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringP2P.v2",
    "BiorxivClusteringS2S",
    "BiorxivClusteringS2S.v2",
    "MedrxivClusteringP2P",
    "MedrxivClusteringP2P.v2",
    "MedrxivClusteringS2S",
    "MedrxivClusteringS2S.v2",
    "TwentyNewsgroupsClustering",
    "TwentyNewsgroupsClustering.v2",
    "StackExchangeClustering",
    "StackExchangeClustering.v2",
    "StackExchangeClusteringP2P",
    "StackExchangeClusteringP2P.v2",
    "RedditClustering",
    "RedditClustering.v2",
    "RedditClusteringP2P",
    "RedditClusteringP2P.v2",
    "STSBenchmark",
    "STSBenchmarkMultilingualSTS",  # translated, not trained on
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MIRACLReranking",
    "MrTidyRetrieval",
}

NV_embed_v2 = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
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
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=nvidia_training_datasets,
    public_training_code=None,
    public_training_data=None,
)

NV_embed_v1 = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
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
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    training_datasets=nvidia_training_datasets,
    public_training_code=None,
    public_training_data=None,
)
