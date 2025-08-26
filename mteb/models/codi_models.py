from __future__ import annotations

import logging
from functools import partial

from mteb.model_meta import ModelMeta
from mteb.encoder_interface import PromptType
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper

logger = logging.getLogger(__name__)

codi_instruction = {
    "CmedqaRetrieval": {
        "query": "Given a Chinese community medical question, retrieve replies that best answer the question",
        "passage": "",
    },
    "CovidRetrieval": {
        "query": "Given a question on COVID-19, retrieve news articles that answer the question",
        "passage": "",
    },
    "DuRetrieval": {
        "query": "Given a Chinese search query, retrieve web passages that answer the question",
        "passage": "",
    },
    "EcomRetrieval": {
        "query": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
        "passage": "",
    },
    "MedicalRetrieval": {
        "query": "Given a medical question, retrieve user replies that best answer the question",
        "passage": "",
    },
    "MMarcoRetrieval": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "passage": "",
    },
    "T2Retrieval": {
        "query": "Given a Chinese search query, retrieve web passages that answer the question",
        "passage": "",
    },
    "VideoRetrieval": {
        "query": "Given a video search query, retrieve the titles of relevant videos",
        "passage": "",
    },
    "AFQMC": "Represent the text in conversations between users and financial customer service, retrieve semantically similar text",
    "ATEC": "Represent the text in conversations between users and financial customer service, retrieve semantically similar text",
    "BQ": "Represent the user problem descriptions when handling bank credit business, retrieve semantically similar text",
    "LCQMC": "Represent the user question descriptions on general question-answering platforms, retrieve semantically similar text",
    "PAWSX": "Represent the Chinese Translations of English Encyclopedias, retrieve semantically similar text",
    "QBQTC": "Represent the web search query, retrieve semantically similar text",
    "STSB": "Represent the short general domain sentences, retrieve semantically similar text",
    "T2Reranking": {
        "query": "Given a Chinese search query, retrieve web passages that answer the question",
        "passage": "",
    },
    "MMarcoReranking": {
        "query": "Given a web search query, retrieve relevant passages that answer the query",
        "passage": "",
    },
    "CMedQAv1-reranking": {
        "query": "Given a Chinese community medical question, retrieve replies that best answer the question",
        "passage": "",
    },
    "CMedQAv2-reranking": {
        "query": "Given a Chinese community medical question, retrieve replies that best answer the question",
        "passage": "",
    },
    "Ocnli": "Retrieve semantically similar text",
    "Cmnli": "Retrieve semantically similar text",
    "TNews": "Classify the fine-grained category of the given news title",
    "IFlyTek": "Given an App description text, find the appropriate fine-grained category",
    "Waimai": "Classify the customer review from a food takeaway platform into positive or negative",
    "OnlineShopping": "Classify the customer review for online shopping into positive or negative",
    "JDReview": "Classify the customer review for iPhone on e-commerce platform into positive or negative",
    "MultilingualSentiment": "Classify sentiment of the customer review into positive, neutral, or negative",
    "CLSClusteringS2S": "Identify the main category of scholar papers based on the titles",
    "CLSClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
    "ThuNewsClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "ThuNewsClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
}


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.passage:
        return "<s>"
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = list(instruction.values())[0]
        else:
            instruction = instruction[prompt_type]
    return f"<s>Instruction: {instruction} \nQuery: "


training_data = {
    "T2Retrieval": ["train"],
    "DuRetrieval": ["train"],
    "T2Reranking": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "BQ": ["train"],
    "LCQMC": ["train"],
    "PAWSX": ["train"],
    "STS-B": ["train"],
    "AFQMC": ["train"],
    "Cmnli": ["train"],
    "Ocnli": ["train"],
}

model_name_or_path = "Youtu-RAG/CoDi-Embedding-V1"

CoDiEmb_Embedding_V1 = ModelMeta(
    name="Youtu-RAG/CoDi-Embedding-V1",
    languages=["zho-Hans"],
    revision="9ee4337715ce337f12b8d30f20e87e8528ccedd6",
    release_date="2025-08-20",
    loader=partial(
        InstructSentenceTransformerWrapper,
        model_name_or_path,
        revision="9ee4337715ce337f12b8d30f20e87e8528ccedd6",
        instruction_template=instruction_template,
        apply_instruction_to_passages=True,
        prompts_dict=codi_instruction,
        trust_remote_code=True,
        max_seq_length=4096,
    ),
    open_weights=True,
    n_parameters=2724880896,
    memory_usage_mb=None,
    embed_dim=2304,
    license="apache-2.0",
    max_tokens=4096,
    reference="https://huggingface.co/Youtu-RAG/CoDi-Embedding-V1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)
