import logging
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_implementations.bge_models import (
    bge_chinese_training_data,
    bge_full_data,
    bge_m3_training_data,
)
from mteb.models.model_implementations.e5_instruct import E5_MISTRAL_TRAINING_DATA
from mteb.models.model_implementations.nvidia_models import nvidia_training_datasets
from mteb.models.model_implementations.qzhou_models import qzhou_training_data
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

jasper_token_compression_600m_prompts_dict = {
    "AFQMC": "Retrieve semantically similar text",
    "AILACasedocs": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "document": "",
    },
    "AILAStatutes": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "document": "",
    },
    "ATEC": "Retrieve semantically similar text",
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
    "ArXivHierarchicalClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArXivHierarchicalClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "ArguAna": {
        "query": "Given a claim, find documents that refute the claim",
        "document": "Given a claim, find documents that refute the claim",
    },
    "AskUbuntuDupQuestions": {
        "query": "Retrieve duplicate questions from AskUbuntu forum",
        "document": "",
    },
    "BIOSSES": "Retrieve semantically similar text",
    "BQ": "Retrieve semantically similar text",
    "Banking77Classification": "Given a online banking query, find the corresponding intents",
    "BiorxivClusteringP2P.v2": "Identify the main category of Biorxiv papers based on the titles and abstracts",
    "CLSClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
    "CLSClusteringS2S": "Identify the main category of scholar papers based on the titles",
    "CMedQAv1-reranking": {
        "query": "Given a Chinese community medical question, retrieve replies that best answer the question",
        "document": "",
    },
    "CMedQAv2-reranking": {
        "query": "Given a Chinese community medical question, retrieve replies that best answer the question",
        "document": "",
    },
    "CQADupstackGamingRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    },
    "CQADupstackUnixRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    },
    "ClimateFEVERHardNegatives": {
        "query": "Given a claim about climate change, retrieve documents that support or refute the claim",
        "document": "",
    },
    "CmedqaRetrieval": {
        "query": "Given a Chinese community medical question, retrieve replies that best answer the question",
        "document": "",
    },
    "Cmnli": "Retrieve semantically similar text.",
    "CovidRetrieval": {
        "query": "Given a question on COVID-19, retrieve news articles that answer the question",
        "document": "",
    },
    "DuRetrieval": {
        "query": "Given a Chinese search query, retrieve web passages that answer the question",
        "document": "",
    },
    "EcomRetrieval": {
        "query": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
        "document": "",
    },
    "FEVERHardNegatives": {
        "query": "Given a claim, retrieve documents that support or refute the claim",
        "document": "",
    },
    "FiQA2018": {
        "query": "Given a financial question, retrieve user replies that best answer the question",
        "document": "",
    },
    "GerDaLIRSmall": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "document": "",
    },
    "HotpotQAHardNegatives": {
        "query": "Given a multi-hop question, retrieve documents that can help answer the question",
        "document": "",
    },
    "IFlyTek": "Given an App description text, find the appropriate fine-grained category",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset",
    "JDReview": "Classify the customer review for iPhone on e-commerce platform into positive or negative",
    "LCQMC": "Retrieve semantically similar text",
    "LeCaRDv2": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "document": "",
    },
    "LegalBenchConsumerContractsQA": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "document": "",
    },
    "LegalBenchCorporateLobbying": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "document": "",
    },
    "LegalQuAD": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "document": "",
    },
    "LegalSummarization": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "document": "",
    },
    "MMarcoReranking": {
        "query": "Given a Chinese search query, retrieve web passages that answer the question",
        "document": "",
    },
    "MMarcoRetrieval": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "document": "",
    },
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios",
    "MedicalRetrieval": {
        "query": "Given a medical question, retrieve user replies that best answer the question",
        "document": "",
    },
    "MedrxivClusteringP2P.v2": "Identify the main category of Medrxiv papers based on the titles and abstracts",
    "MedrxivClusteringS2S.v2": "Identify the main category of Medrxiv papers based on the titles",
    "MindSmallReranking": {
        "query": "Retrieve relevant news articles based on user browsing history",
        "document": "",
    },
    "MultilingualSentiment": "Classify sentiment of the customer review into positive, neutral, or negative",
    "Ocnli": "Retrieve semantically similar text.",
    "OnlineShopping": "Classify the customer review for online shopping into positive or negative",
    "PAWSX": "Retrieve semantically similar text",
    "QBQTC": "Retrieve semantically similar text",
    "SCIDOCS": {
        "query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
        "document": "",
    },
    "SICK-R": "Retrieve semantically similar text",
    "STS12": "Retrieve semantically similar text",
    "STS13": "Retrieve semantically similar text",
    "STS14": "Retrieve semantically similar text",
    "STS15": "Retrieve semantically similar text",
    "STS17": "Retrieve semantically similar text",
    "STS22.v2": "Retrieve semantically similar text",
    "STSB": "Retrieve semantically similar text",
    "STSBenchmark": "Retrieve semantically similar text",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum",
    "StackExchangeClustering.v2": "Identify the topic or theme of StackExchange posts based on the titles",
    "StackExchangeClusteringP2P.v2": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    "SummEvalSummarization.v2": "Retrieve semantically similar text",
    "T2Reranking": {
        "query": "Given a Chinese search query, retrieve web passages that answer the question",
        "document": "",
    },
    "T2Retrieval": {
        "query": "Given a Chinese search query, retrieve web passages that answer the question",
        "document": "",
    },
    "TNews": "Classify the fine-grained category of the given news title",
    "TRECCOVID": {
        "query": "Given a query on COVID-19, retrieve documents that answer the query",
        "document": "",
    },
    "ThuNewsClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    "ThuNewsClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "Touche2020Retrieval.v3": {
        "query": "Given a question, retrieve detailed and persuasive arguments that answer the question",
        "document": "",
    },
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
    "TwentyNewsgroupsClustering.v2": "Identify the topic or theme of the given news articles",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet",
    "VideoRetrieval": {
        "query": "Given a video search query, retrieve the titles of relevant videos",
        "document": "",
    },
    "Waimai": "Classify the customer review from a food takeaway platform into positive or negative",
}
jasper_token_compression_600m_loader_kwargs = dict(
    model_kwargs={
        "attn_implementation": "sdpa",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True,
    },
    tokenizer_kwargs={"padding_side": "left"},
    trust_remote_code=True,
    prompts_dict=jasper_token_compression_600m_prompts_dict,
    apply_instruction_to_passages=True,
    instruction_template="Instruct: {instruction}\nQuery: ",
    max_seq_length=1024,
)


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


class JasperModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        instruction_template: str | Callable[[str], str] | None = None,
        max_seq_length: int = 2048,
        **kwargs: Any,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.instruction_template = instruction_template
        self.model.max_seq_length = max_seq_length

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        instruction = self.get_task_instruction(task_metadata, prompt_type)

        # to passage prompts won't be applied to passages
        if prompt_type == PromptType.document:
            instruction = None
        inputs = [text for batch in inputs for text in batch["text"]]

        embeddings = self.model.encode(
            inputs,
            normalize_embeddings=True,
            prompt=instruction,
            **kwargs,
        )

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings


jasper_en_v1 = ModelMeta(
    loader=JasperModel,
    loader_kwargs=dict(
        config_kwargs={"is_text_encoder": True, "vector_dim": 12288},
        model_kwargs={
            "attn_implementation": "sdpa",
            "torch_dtype": torch.bfloat16,
        },
        trust_remote_code=True,
        max_seq_length=2048,
        instruction_template="Instruct: {instruction}\nQuery: ",
    ),
    name="NovaSearch/jasper_en_vision_language_v1",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="d6330ce98f8a0d741e781df845904c9484f00efa",
    release_date="2024-12-11",  # first commit
    n_parameters=1_999_000_000,
    memory_usage_mb=3802,
    max_tokens=131072,
    embed_dim=8960,
    license="apache-2.0",
    reference="https://huggingface.co/infgrad/jasper_en_vision_language_v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    training_datasets=set(
        # stage 1, 2, 3
        #  "In jasper model the teacher model is nvidia/NV-Embed-v2", source https://huggingface.co/NovaSearch/jasper_en_vision_language_v1
        # fineweb-edu
        # https://huggingface.co/datasets/sentence-transformers/embedding-training-data
        # stage 4
        # BAAI/Infinity-MM
    )
    | nvidia_training_datasets,
    # training logs https://api.wandb.ai/links/dunnzhang0/z8jqoqpb
    # more codes https://huggingface.co/NovaSearch/jasper_en_vision_language_v1/commit/da9b77d56c23d9398fa8f93af449102784f74e1d
    public_training_code="https://github.com/NovaSearch-Team/RAG-Retrieval/blob/c40f4638b705eb77d88305d2056901ed550f9f4b/rag_retrieval/train/embedding/README.md",
    public_training_data="https://huggingface.co/datasets/infgrad/jasper_text_distill_dataset",
    citation="""
@misc{zhang2025jasperstelladistillationsota,
      title={Jasper and Stella: distillation of SOTA embedding models},
      author={Dun Zhang and Jiacheng Li and Ziyang Zeng and Fulong Wang},
      year={2025},
      eprint={2412.19048},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2412.19048},
}
""",
)


Jasper_Token_Compression_600M = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=jasper_token_compression_600m_loader_kwargs,
    name="infgrad/Jasper-Token-Compression-600M",
    model_type=["dense"],
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    revision="06a100f753a5a96d9e583b3af79c6fcdfacc4719",
    release_date="2025-11-14",
    n_parameters=595776512,
    memory_usage_mb=2272,
    embed_dim=2048,
    license="mit",
    max_tokens=32768,
    reference="https://huggingface.co/infgrad/Jasper-Token-Compression-600M",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/DunZhang/Jasper-Token-Compression-Training",
    # public_training_data: unsupervised data for distillation
    public_training_data="https://huggingface.co/datasets/infgrad/jasper_text_distill_dataset",
    training_datasets=bge_m3_training_data
    | bge_chinese_training_data
    | bge_full_data
    | E5_MISTRAL_TRAINING_DATA
    | qzhou_training_data,
    citation="""
@misc{zhang2025jaspertokencompression600mtechnicalreport,
      title={Jasper-Token-Compression-600M Technical Report},
      author={Dun Zhang and Ziyang Zeng and Yudong Zhou and Shuyang Lu},
      year={2025},
      eprint={2511.14405},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2511.14405},
}
""",
)
