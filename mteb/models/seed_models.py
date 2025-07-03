from __future__ import annotations

import logging
import time
from functools import partial
from typing import Any

import numpy as np
import torch
import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.bge_models import bge_chinese_training_data
from mteb.models.nvidia_models import nvidia_training_datasets
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


seed_1_5_training_data = (
    {
        "PAWSX": ["train"],
        "QBQTC": ["train"],
        "STSB": ["train"],
        "TNews": ["train"],
        "Waimai": ["train"],
        "IFlyTek": ["train"],
    }
    | bge_chinese_training_data
    | nvidia_training_datasets
)


class SeedWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        tokenizer_name: str = "cl100k_base",
        embed_dim: int | None = None,
        available_embed_dims: list[int | None] = [None],
        **kwargs,
    ) -> None:
        """Wrapper for Seed embedding API."""
        requires_package(
            self,
            "volcenginesdkarkruntime",
            "ByteDance Seed",
            "pip install mteb[ark]",
        )
        from volcenginesdkarkruntime import Ark

        requires_package(
            self,
            "tiktoken",
            "ByteDance Seed",
            "pip install mteb[ark]",
        )
        import tiktoken

        self._client = Ark()
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._embed_dim = embed_dim
        self._available_embed_dims = available_embed_dims
        self._encoding = tiktoken.get_encoding(tokenizer_name)

    def truncate_text_tokens(self, text):
        """Truncate a string to have `max_tokens` according to the given encoding."""
        truncated_sentence = self._encoding.encode(text)[: self._max_tokens]
        return self._encoding.decode(truncated_sentence)

    def _encode(
        self, inputs: list[str], task_name: str, prompt_type: PromptType | None = None
    ):
        assert (
            self._embed_dim is None or self._embed_dim in self._available_embed_dims
        ), (
            f"Available embed_dims are {self._available_embed_dims}, found {self._embed_dim}"
        )

        if prompt_type == PromptType("query") or prompt_type is None:
            if task_name in TASK_NAME_TO_INSTRUCTION:
                instruction = TASK_NAME_TO_INSTRUCTION[task_name]
            else:
                instruction = DEFAULT_INSTRUCTION
            inputs = [instruction + i for i in inputs]

        response = self._client.embeddings.create(
            input=inputs,
            model=self._model_name,
            encoding_format="float",
        )
        outputs = torch.tensor(
            [d.embedding for d in response.data], dtype=torch.bfloat16
        )
        if self._embed_dim is not None:
            outputs = outputs[:, : self._embed_dim]
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        return outputs.float().tolist()

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        retries: int = 5,
        **kwargs: Any,
    ) -> np.ndarray:
        trimmed_sentences = []
        for sentence in sentences:
            encoded_sentence = self._encoding.encode(sentence)
            if len(encoded_sentence) > self._max_tokens:
                truncated_sentence = self.truncate_text_tokens(sentence)
                trimmed_sentences.append(truncated_sentence)
            else:
                trimmed_sentences.append(sentence)

        max_batch_size = kwargs.get("batch_size", 32)
        sublists = [
            trimmed_sentences[i : i + max_batch_size]
            for i in range(0, len(trimmed_sentences), max_batch_size)
        ]

        show_progress_bar = (
            False
            if "show_progress_bar" not in kwargs
            else kwargs.pop("show_progress_bar")
        )

        all_embeddings = []

        for i, sublist in enumerate(
            tqdm.tqdm(sublists, leave=False, disable=not show_progress_bar)
        ):
            while retries > 0:
                try:
                    embedding = self._encode(sublist, task_name, prompt_type)
                    all_embeddings.extend(embedding)
                    break
                except Exception as e:
                    # Sleep due to too many requests
                    time.sleep(1)
                    logger.warning(
                        f"Retrying... {retries} retries left. Error: {str(e)}"
                    )
                    retries -= 1
                    if retries == 0:
                        raise e

        return np.array(all_embeddings)


TASK_NAME_TO_INSTRUCTION = {
    "ArguAna": "Instruct: Given a claim, find documents that refute the claim\nQuery: ",
    "ClimateFEVERHardNegatives": "Instruct: Given a claim about climate change, retrieve documents that support or refute the claim\nQuery: ",
    "FEVERHardNegatives": "Instruct: Given a claim, retrieve documents that support or refute the claim\nQuery: ",
    "FiQA2018": "Instruct: Given a financial question, retrieve user replies that best answer the question\nQuery: ",
    "HotpotQAHardNegatives": "Instruct: Given a multi-hop question, retrieve documents that can help answer the question\nQuery: ",
    "SCIDOCS": "Instruct: Given a scientific paper title, retrieve paper abstracts that are cited by the given paper\nQuery: ",
    "Touche2020Retrieval.v3": "Instruct: Given a question, retrieve detailed and persuasive arguments that answer the question\nQuery: ",
    "TRECCOVID": "Instruct: Given a query on COVID-19, retrieve documents that answer the query\nQuery: ",
    "AskUbuntuDupQuestions": "Instruct: Retrieve duplicate questions from AskUbuntu forum\nQuery: ",
    "MindSmallReranking": "Instruct: Retrieve relevant news articles based on user browsing history\nQuery: ",
    "SprintDuplicateQuestions": "Instruct: Retrieve duplicate questions from Sprint forum\nQuery: ",
    "TwitterSemEval2015": "Instruct: Retrieve tweets that are semantically similar to the given tweet\nQuery: ",
    "TwitterURLCorpus": "Instruct: Retrieve tweets that are semantically similar to the given tweet\nQuery: ",
    "CQADupstackGamingRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "CQADupstackUnixRetrieval": "Instruct: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\nQuery: ",
    "DuRetrieval": "Instruct: Given a Chinese search query, retrieve web passages that answer the question\nQuery: ",
    "T2Retrieval": "Instruct: Given a Chinese search query, retrieve web passages that answer the question\nQuery: ",
    "MMarcoRetrieval": "Instruct: Given a Chinese search query, retrieve web passages that answer the question\nQuery: ",
    "MMarcoReranking": "Instruct: Based on a Chinese search question, evaluate and rank the web passages that provide answers to the question\nQuery: ",
    "T2Reranking": "Instruct: Based on a Chinese search question, evaluate and rank the web passages that provide answers to the question\nQuery: ",
    "CMedQAv1-reranking": "Instruct: Based a Chinese medical question, evaluate and rank the medical information that provide answers to the question\nQuery: ",
    "CMedQAv2-reranking": "Instruct: Based a Chinese medical question, evaluate and rank the medical information that provide answers to the question\nQuery: ",
    "CovidRetrieval": "Instruct: Given a Chinese question on COVID-19, retrieve news articles that answer the question\nQuery: ",
    "CmedqaRetrieval": "Instruct: Given a Chinese query on medical information, retrieve relevant documents that answer the query\nQuery: ",
    "VideoRetrieval": "Instruct: Given a video search query, retrieve the titles of relevant videos\nQuery: ",
    "EcomRetrieval": "Instruct: Given a user query from an e-commerce website, retrieve description sentences of relevant products\nQuery: ",
    "MedicalRetrieval": "Instruct: Given a medical question, retrieve user replies that best answer the question\nQuery: ",
    "ATEC": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "BQ": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "LCQMC": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "PAWSX": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "STSB": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "AFQMC": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "QBQTC": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "STS22.v2": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "BIOSSES": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "SICK-R": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "STS12": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "STS13": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "STS14": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "STS15": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "STS17": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "STSBenchmark": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "SummEvalSummarization.v2": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "Ocnli": "Instruct: Given a premise, retrieve a hypothesis that is entailed by the premise\nQuery: ",
    "Cmnli": "Instruct: Given a premise, retrieve a hypothesis that is entailed by the premise\nQuery: ",
    "TNews": "Instruct: Classify the fine-grained category of the given news title\nQuery: ",
    "IFlyTek": "Instruct: Given an App description text, find the appropriate fine-grained category\nQuery: ",
    "MultilingualSentiment": "Instruct: Classify sentiment of the customer review into positive, neutral, or negative\nQuery: ",
    "JDReview": "Instruct: Classify the customer review for iPhone on e-commerce platform into positive or negative\nQuery: ",
    "OnlineShopping": "Instruct: Classify the customer review for online shopping into positive or negative\nQuery: ",
    "Waimai": "Instruct: Classify the customer review from a food takeaway platform into positive or negative\nQuery: ",
    "AmazonCounterfactualClassification": "Instruct: Classify a given Amazon customer review text as either counterfactual or not-counterfactual\nQuery: ",
    "AmazonReviewsClassification": "Instruct: Classify a given Amazon review into its appropriate rating category\nQuery: ",
    "Banking77Classification": "Instruct: Given an online banking query, find the corresponding intents\nQuery: ",
    "ImdbClassification": "Instruct: Classify the sentiment expressed in a given movie review text from the IMDB dataset\nQuery: ",
    "MassiveIntentClassification": "Instruct: Given a user utterance as query, find the user intents\nQuery: ",
    "MassiveScenarioClassification": "Instruct: Given a user utterance as query, find the user scenarios\nQuery: ",
    "MTOPDomainClassification": "Instruct: Classify the intent domain of a given utterance in task-oriented conversation\nQuery: ",
    "ToxicConversationsClassification": "Instruct: Classify the given comments as either toxic or not toxic\nQuery: ",
    "TweetSentimentExtractionClassification": "Instruct: Classify the sentiment of a given tweet as either positive, negative, or neutral\nQuery: ",
    "ArXivHierarchicalClusteringP2P": "Instruct: Identify the main and secondary category of Arxiv papers based on the titles and abstracts\nQuery: ",
    "ArXivHierarchicalClusteringS2S": "Instruct: Identify the main and secondary category of Arxiv papers based on the titles\nQuery: ",
    "BiorxivClusteringP2P.v2": "Instruct: Identify the main category of Biorxiv papers based on the titles and abstracts\nQuery: ",
    "MedrxivClusteringP2P.v2": "Instruct: Identify the main category of Medrxiv papers based on the titles and abstracts\nQuery: ",
    "MedrxivClusteringS2S.v2": "Instruct: Identify the main category of Medrxiv papers based on the titles\nQuery: ",
    "StackExchangeClustering.v2": "Instruct: Identify the topic or theme of StackExchange posts based on the titles\nQuery: ",
    "StackExchangeClusteringP2P.v2": "Instruct: Identify the topic or theme of StackExchange posts based on the given paragraphs\nQuery: ",
    "TwentyNewsgroupsClustering.v2": "Instruct: Identify the topic or theme of the given news articles\nQuery: ",
    "CLSClusteringS2S": "Instruct: Identify the main category of scholar papers based on the titles\nQuery: ",
    "CLSClusteringP2P": "Instruct: Identify the main category of scholar papers based on the titles and abstracts\nQuery: ",
    "ThuNewsClusteringS2S": "Instruct: Identify the topic or theme of the given news articles based on the titles\nQuery: ",
    "ThuNewsClusteringP2P": "Instruct: Identify the topic or theme of the given news articles based on the titles and abstracts\nQuery: ",
}

DEFAULT_INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "


seed_embedding = ModelMeta(
    name="ByteDance-Seed/Seed1.5-Embedding",
    revision="4",
    release_date="2025-04-25",
    languages=[
        "eng-Latn",
        "zho-Hans",
    ],
    loader=partial(
        SeedWrapper,
        model_name="doubao-embedding-large-text-250515",
        max_tokens=32000,  # tiktoken
        available_embed_dims=[2048, 1024, 512, 256],
    ),
    max_tokens=32768,
    embed_dim=2048,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://seed1-5-embedding.github.io/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    training_datasets=seed_1_5_training_data,
    public_training_code=None,
    public_training_data=None,
)
