from __future__ import annotations

import os
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
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed



def multimodal_embedding(image_url=None, text_content=None):
    auth_token = os.getenv("VOLCES_AUTH_TOKEN")
    model_name = "doubao-embedding-vision-250615"
    api_url = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "x-ark-vlm1": "true", 
        "Content-Type": "application/json"
    }

    if image_url is not None and text_content is None:
        inputs = []
        for image in image_url:
            image_format = "jpeg"
            image_data = f"data:image/{image_format};base64,{image}"
            inputs.append({
                "type": "image_url",
                "image_url": {"url": image_data}
            })

        payload = {
            "model": model_name,
            "input": inputs
        }
    elif image_url is None and text_content is not None:
        payload = {
            "model": model_name,
            "input": [
                {
                    "type": "text",
                    "text": text_content
                },
            ]
        }
    else:
        inputs = []
        for image in image_url:
            image_format = "jpeg"
            image_data = f"data:image/{image_format};base64,{image}"
            inputs.append({
                "type": "image_url",
                "image_url": {"url": image_data}
            })
        inputs.append({
            "type": "text",
            "text": text_content
        })
        payload = {
            "model": model_name,
            "input": inputs
        }

    try:
        response = requests.post(
            url=api_url,
            headers=headers,
            json=payload,
            timeout=10
        )

        # 检查HTTP状态码
        response.raise_for_status()

        # 尝试解析JSON响应
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP错误 ({http_err.response.status_code}): {http_err}")
    except requests.exceptions.JSONDecodeError:
        print("错误：响应不是有效的JSON格式")
        print(f"原始响应：{response.text}")
    except requests.exceptions.Timeout:
        print("错误：请求超时")
    except Exception as e:
        print(f"未知错误：{str(e)}")

    return None


def multi_thread_encode(sentences, batch_size=1, max_workers=8):
    batches = []
    for idx in range(0, len(sentences), batch_size):
        batches.append((idx // batch_size, sentences[idx:idx + batch_size]))

    n_batches = len(batches)
    results = [None] * n_batches  # Pre-allocated result list
    all_embeddings = []           # Final ordered embeddings

    def _process_batch(batch_idx, batch_sentences):
        sentence = batch_sentences[0]
        
        retries = 5
        while retries > 0:
            try:
                resp = multimodal_embedding(text_content=sentence)
                embedding = torch.tensor(resp["data"]["embedding"])
                break
            except Exception as e:
                time.sleep(1)
                logger.warning(
                    f"Retrying... {retries} retries left. Error: {str(e)}"
                )
                retries -= 1
                if retries == 0:
                    raise e
        return batch_idx, embedding

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_batch, idx, batch): idx
            for idx, batch in batches
        }

        for future in as_completed(futures):
            batch_idx, embeddings = future.result()
            results[batch_idx] = embeddings

    for batch_embeddings in results:
        all_embeddings.append(batch_embeddings)

    all_embeddings = torch.stack(all_embeddings, dim=0)
    all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=-1)

    return all_embeddings.float().cpu()




logger = logging.getLogger(__name__)


doubao_embedding_training_data = (
    {
        "PAWSX": ["train"],
        "QBQTC": ["train"],
        "STSB": ["train"],
        "TNews": ["train"],
        "Waimai": ["train"],
        "IFlyTek": ["train"],
    }
    | bge_chinese_training_data
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
            "pip install mteb[ark]",
            "tiktoken",
        )
        import tiktoken

        self._model_name = model_name
        self._max_tokens = max_tokens
        self._embed_dim = embed_dim
        self._available_embed_dims = available_embed_dims
        self._encoding = tiktoken.get_encoding(tokenizer_name)

    def truncate_text_tokens(self, text):
        """Truncate a string to have `max_tokens` according to the given encoding."""
        truncated_sentence = self._encoding.encode(text)[: self._max_tokens]
        return self._encoding.decode(truncated_sentence)

    def _encode(self, inputs: list[str], task_name: str, prompt_type: PromptType | None = None):
        assert (
            self._embed_dim is None or self._embed_dim in self._available_embed_dims
        ), (
            f"Available embed_dims are {self._available_embed_dims}, found {self._embed_dim}"
        )

        if prompt_type == PromptType("query") or prompt_type is None:
            if task_name in TASK_NAME_TO_INSTRUCTION:
                instruction = TASK_NAME_TO_INSTRUCTION[task_name]
            else:
                raise ValueError(f"Unknown task_name {task_name}")
            inputs = [instruction.format(i) for i in inputs]
        else:
            pass
        
        outputs = multi_thread_encode(inputs)

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

        max_batch_size = kwargs.get("batch_size", 256)
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

        for sublist in tqdm.tqdm(sublists, leave=False, disable=not show_progress_bar):
            embedding = self._encode(sublist, task_name, prompt_type)
            all_embeddings.extend(embedding)

        return np.array(all_embeddings)

   

TASK_NAME_TO_INSTRUCTION = {
    "ArguAna": "Given a claim, find documents that refute the claim\n{}",
    "ClimateFEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim\n{}",
    "FEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim\n{}",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question\n{}",
    "HotpotQAHardNegatives": "Given a multi-hop question, retrieve documents that can help answer the question\n{}",
    "SCIDOCS": "Given a title of a scientific paper, retrieve the titles of other relevant papers\n{}",
    "Touche2020Retrieval.v3": "Given a question, retrieve detailed and persuasive arguments that answer the question\n{}",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query\n{}",
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum\n{}",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history\n{}",
    "SprintDuplicateQuestions": "Retrieve semantically similar text\n{}",
    "TwitterSemEval2015": "Retrieve semantically similar text\n{}",
    "TwitterURLCorpus": "Retrieve semantically similar text\n{}",
    "CQADupstackGamingRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given questionn\n{}",
    "CQADupstackUnixRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\n{}",
    "DuRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "T2Retrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "MMarcoRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "MMarcoReranking": "为这个句子生成表示以用于检索相关内容：{}",
    "T2Reranking": "为这个句子生成表示以用于检索相关内容：{}",
    "CMedQAv1-reranking": "为这个句子生成表示以用于检索相关内容：{}",
    "CMedQAv2-reranking": "为这个句子生成表示以用于检索相关内容：{}",
    "CovidRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "CmedqaRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "VideoRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "EcomRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "MedicalRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "ATEC": "Retrieve semantically similar text\n{}",
    "BQ": "Retrieve semantically similar text\n{}",
    "LCQMC": "Retrieve semantically similar text\n{}",
    "PAWSX": "Retrieve semantically similar text\n{}",
    "STSB": "Retrieve semantically similar text\n{}",
    "AFQMC": "Retrieve semantically similar text\n{}",
    "QBQTC": "Retrieve semantically similar text\n{}",
    "STS22.v2": "Retrieve semantically similar text\n{}",
    "BIOSSES": "Retrieve semantically similar text\n{}",
    "SICK-R": "Retrieve semantically similar text\n{}",
    "STS12": "Retrieve semantically similar text\n{}",
    "STS13": "Retrieve semantically similar text\n{}",
    "STS14": "Retrieve semantically similar text\n{}",
    "STS15": "Retrieve semantically similar text\n{}",
    "STS17": "Retrieve semantically similar text\n{}",
    "STSBenchmark": "Retrieve semantically similar text\n{}",
    "SummEvalSummarization.v2": "Retrieve semantically similar text\n{}",
    "Ocnli": "Retrieve semantically similar text\n{}",
    "Cmnli": "Retrieve semantically similar text\n{}",
    "TNews": "Classify the fine-grained category of the given news title\n{}",
    "IFlyTek": "Given an App description text, find the appropriate fine-grained category\n{}",
    "MultilingualSentiment": "Classify sentiment of the customer review into positive, neutral, or negative\n{}",
    "JDReview": "Classify the customer review for iPhone on e-commerce platform into positive or negative\n{}",
    "OnlineShopping": "Classify the customer review for online shopping into positive or negative\n{}",
    "Waimai": "Classify the customer review from a food takeaway platform into positive or negative\n{}",
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual\n{}",
    "Banking77Classification": "Given a online banking query, find the corresponding intents\n{}",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset\n{}",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents\n{}",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios\n{}",
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation\n{}",
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic\n{}",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral\n{}",
    "ArXivHierarchicalClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts\n{}",
    "ArXivHierarchicalClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles\n{}",
    "BiorxivClusteringP2P.v2": "Identify the main category of Biorxiv papers based on the titles and abstracts\n{}",
    "MedrxivClusteringP2P.v2": "Identify the main category of Medrxiv papers based on the titles and abstracts\n{}",
    "MedrxivClusteringS2S.v2": "Identify the main category of Medrxiv papers based on the titles\n{}",
    "StackExchangeClustering.v2": "Identify the topic or theme of StackExchange posts based on the titles\n{}",
    "StackExchangeClusteringP2P.v2": "Identify the topic or theme of StackExchange posts based on the given paragraphs\n{}",
    "TwentyNewsgroupsClustering.v2": "Identify the topic or theme of the given news articles\n{}",
    "CLSClusteringS2S": "Identify the main category of scholar papers based on the titles\n{}",
    "CLSClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts\n{}",
    "ThuNewsClusteringS2S": "Identify the topic or theme of the given news articles based on the titles\n{}",
    "ThuNewsClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents\n{}",
}


seed_embedding = ModelMeta(
    name="Bytedance/Seed-1.6-embedding",
    revision="1",
    release_date="2025-06-18",
    languages=[
        "eng-Latn",
        "zho-Hans",
    ],
    loader=partial(
        SeedWrapper,
        model_name="Bytedance/Seed-1.6-embedding",
        max_tokens=32000,
        available_embed_dims=[2048, 1024, 512],
    ),
    max_tokens=32768,
    embed_dim=2048,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://console.volcengine.com/ark/region:ark+cn-beijing/model/detail?Id=doubao-embedding-vision&projectName=default",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    training_datasets=doubao_embedding_training_data,
    public_training_code=None,
    public_training_data=None,
)
