import hashlib
import json
import logging
import os
import random
import string
import time
from typing import Any

import numpy as np
import requests
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, BatchedInput, PromptType

from .bge_models import bge_full_data
from .e5_instruct import E5_MISTRAL_TRAINING_DATA

conan_zh_datasets = {
    "BQ",
    "LCQMC",
    "PAWSX",
    "STS-B",
    "DuRetrieval",
    "AFQMC",
    "Cmnli",
    "Ocnli",
    "T2Retrieval",
    "T2Reranking",
    "MMarcoReranking",
    "CMedQAv2-reranking",
}

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, qps, max_retries=3):
        self.qps = qps
        self.min_interval = 1.0 / qps
        self.last_request_time = 0
        self.max_retries = max_retries

    def wait(self):
        """Simple rate limiting logic"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic

        Args:
            func: The function to execute
            *args: Arguments to pass to the function
            **kwargs: Arguments to pass to the function

        Returns:
            The result of the function execution
        """
        retries = 0
        while retries < self.max_retries:
            self.wait()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries < self.max_retries:
                    sleep_time = 10 * retries
                    logger.warning(
                        f"Request failed (attempt {retries}/{self.max_retries}), "
                        f"sleeping for {sleep_time}s. Error: {str(e)}"
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Max retries reached. Last error: {str(e)}")
                    raise


class Client:
    def __init__(self, ak, sk, url, timeout=30):
        self.ak = ak
        self.sk = sk
        self.url = url
        self.timeout = timeout
        self.rate_limiter = RateLimiter(qps=5, max_retries=3)

    def _random_password(self, size=40, chars=None):
        if chars is None:
            chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
        random_chars = random.SystemRandom().choice
        return "".join(random_chars(chars) for _ in range(size))

    def __signature(self, random_str, time_stamp):
        params_str = f"{self.ak}:{time_stamp}:{random_str}:{self.sk}"
        encoded_params_str = params_str.encode("utf-8")
        return hashlib.md5(encoded_params_str).hexdigest()

    def get_signature(self):
        timestamp = int(time.time())
        random_str = self._random_password(20)
        sig = self.__signature(random_str, timestamp)
        params = {
            "timestamp": timestamp,
            "random": random_str,
            "app_id": self.ak,
            "sign": sig,
        }
        return params

    def _do_request(self, text):
        """Execute the actual request without retry logic"""
        params = self.get_signature()
        params["body"] = text
        params["content_id"] = f"test_{int(time.time())}"
        headers = {"Content-Type": "application/json"}

        rsp = requests.post(
            self.url, data=json.dumps(params), timeout=self.timeout, headers=headers
        )
        result = rsp.json()

        if rsp.status_code != 200:
            raise Exception(
                f"API request failed with status {rsp.status_code}: {result}"
            )

        return result

    def embed(self, text):
        """Embed text using the server with rate limiting and retry logic

        Args:
            text: The input text to embed

        Returns:
            dict: Response containing embedding
        """
        # Use rate_limiter to execute the request, handling rate limiting and retries
        return self.rate_limiter.execute_with_retry(self._do_request, text)


class ConanWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        api_model_name: str | None = None,
        **kwargs,
    ) -> None:
        ak = os.getenv("CONAN_AK")
        sk = os.getenv("CONAN_SK")
        if not ak or not sk:
            raise ValueError("CONAN_AK and CONAN_SK environment variables must be set")

        self.client = Client(ak=ak, sk=sk, url="https://ai.om.qq.com/api/conan/v2")
        self.model_name = api_model_name

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
        embeddings = []
        sentences = [text for batch in inputs for text in batch["text"]]

        for sentence in sentences:
            try:
                result = self.client.embed(sentence)
                if "embedding" not in result:
                    raise ValueError(f"No embedding in response: {result}")
                embeddings.append(result["embedding"])
            except Exception as e:
                logger.error(f"Failed to embed sentence: {str(e)}")
                raise

        return np.array(embeddings)


Conan_embedding_v2 = ModelMeta(
    name="TencentBAC/Conan-embedding-v2",
    revision="e5c87c63889630bca87486f6a2645ed97c5ddb17",
    release_date="2025-04-10",
    languages=[
        "eng-Latn",
        "zho-Hans",
    ],
    loader=ConanWrapper,
    loader_kwargs=dict(
        api_model_name="Conan-embedding-v2",
    ),
    max_tokens=32768,
    embed_dim=3584,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license="apache-2.0",
    reference="https://huggingface.co/TencentBAC/Conan-embedding-v2",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    training_datasets=E5_MISTRAL_TRAINING_DATA | bge_full_data | conan_zh_datasets,
    public_training_code=None,
    public_training_data=None,
)
