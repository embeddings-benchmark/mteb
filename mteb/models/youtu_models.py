from __future__ import annotations

import logging
from functools import partial
import os
import json
import time
import types
import numpy as np
import torch
import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


youtu_instruction = {
    "CmedqaRetrieval": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "CovidRetrieval": "Given a question on COVID-19, retrieve news articles that answer the question",
    "DuRetrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "EcomRetrieval": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
    "MedicalRetrieval": "Given a medical question, retrieve user replies that best answer the question",
    "MMarcoRetrieval": "Given a web search query, retrieve relevant passages that answer the query",
    "T2Retrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "VideoRetrieval": "Given a video search query, retrieve the titles of relevant videos",
    "AFQMC": "Represent the text in conversations between users and financial customer service, retrieve semantically similar text",
    "ATEC": "Represent the text in conversations between users and financial customer service, retrieve semantically similar text",
    "BQ": "Represent the user problem descriptions when handling bank credit business, retrieve semantically similar text",
    "LCQMC": "Represent the user question descriptions on general question-answering platforms, retrieve semantically similar text",
    "PAWSX": "Represent the Chinese Translations of English Encyclopedias, retrieve semantically similar text",
    "QBQTC": "Represent the web search query, retrieve semantically similar text",
    "STSB": "Represent the short general domain sentences, retrieve semantically similar text",
    "T2Reranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "MMarcoReranking": "Given a web search query, retrieve relevant passages that answer the query",
    "CMedQAv1-reranking": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "CMedQAv2-reranking": "Given a Chinese community medical question, retrieve replies that best answer the question",
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


training_data = {
    "T2Retrieval": ["train"],
    "DuRetrieval": ["train"],
    "T2Reranking": ["train"],
    "MMarcoReranking": ["train"],
    "CmedqaRetrieval": ["train"],
    "CMedQAv1-reranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "BQ": ["train"],
    "LCQMC": ["train"],
    "PAWSX": ["train"],
    "STS-B": ["train"],
    "AFQMC": ["train"],
    "Cmnli": ["train"],
    "Ocnli": ["train"],
}

class YoutuEmbeddingWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        **kwargs,
    ) -> None:
        requires_package(
            self,
            "tencentcloud.common",
            "tencentcloud.lkeap",
        )

        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
        from tencentcloud.lkeap.v20240522 import lkeap_client, models

        secret_id = os.getenv("TENCENTCLOUD_SECRET_ID")
        secret_key = os.getenv("TENCENTCLOUD_SECRET_KEY")
        if not secret_id or not secret_key:
            raise ValueError("TENCENTCLOUD_SECRET_ID and TENCENTCLOUD_SECRET_KEY environment variables must be set")
        cred = credential.Credential(secret_id, secret_key)

        httpProfile = HttpProfile()
        httpProfile.endpoint = "lkeap.test.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile

        self.client = lkeap_client.LkeapClient(cred, "ap-guangzhou", clientProfile)
        self.models = models
        self.model_name = model_name

    def _encode(
        self, inputs: list[str], task_name: str, prompt_type: PromptType | None = None
    ):
        instruction = ""
        if prompt_type == PromptType("query") or prompt_type is None:
            instruction = youtu_instruction.get(task_name, "")
            instruction = f"Instruction: {instruction} \nQuery: "

        params = {
            "Model": self.model_name,
            "Inputs": inputs,
            "Instruction": instruction
        }
        
        req = self.models.GetEmbeddingRequest()
        req.from_json_string(json.dumps(params))

        resp = self.client.GetEmbedding(req)
        resp = json.loads(resp.to_json_string())
        outputs =[item["Embedding"] for item in resp["Data"]]

        return outputs

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        retries: int = 5,
        **kwargs: Any,
    ) -> np.ndarray:
        max_batch_size = min(4, kwargs.get("batch_size", 4))
        sublists = [
            sentences[i : i + max_batch_size]
            for i in range(0, len(sentences), max_batch_size)
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
            embedding = self._encode(sublist, task_name, prompt_type)
            all_embeddings.extend(embedding)
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


Youtu_Embedding_V1 = ModelMeta(
    name="Youtu-RAG/Youtu-Embedding-V1",
    languages=["zho-Hans"],
    revision="1",
    release_date="2025-09-02",
    loader=partial(
        YoutuEmbeddingWrapper,
        model_name="youtu-embedding-v1"
    ),
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    embed_dim=2304,
    license=None,
    max_tokens=4096,
    reference="https://youtu-embedding-v1.github.io/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)
