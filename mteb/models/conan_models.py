# -*- coding: utf-8 -*-
import os
import hashlib
import random
import string
import time
import requests
import json
import logging

import numpy as np

from functools import partial, wraps
from typing import Any, Literal
from mteb.model_meta import ModelMeta
from mteb.encoder_interface import PromptType
from mteb.models.wrapper import Wrapper
from mteb.models.e5_instruct import E5_MISTRAL_TRAINING_DATA
from mteb.models.bge_models import bge_full_data

conan_zh_datasets = {
    "BQ": ["train"],
    "LCQMC": ["train"],
    "PAWSX": ["train"],
    "STS-B": ["train"],
    "DuRetrieval": ["train"],
    "AFQMC": ["train"],
    "Cmnli": ["train"],
    "Ocnli": ["train"],
    "T2Retrieval": ["train"],
    "T2Reranking": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
}

logger = logging.getLogger(__name__)

class Client:
    def __init__(self, ak, sk, url, timeout=30):
        """
        Initialize the Client

        Args:
            ak: Access Key
            sk: Secret Key
            url: Server URL
            timeout: Request timeout in seconds (default: 30)
        """
        self.ak = ak
        self.sk = sk
        self.url = url
        self.timeout = timeout

    def __random_password(self, size=40, chars=None):
        if chars is None:
            chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
        random_chars = random.SystemRandom().choice
        return "".join(random_chars(chars) for _ in range(size))

    def __signature(self, random_str, time_stamp):
        params_str = "%s:%d:%s:%s" % (self.ak, time_stamp, random_str, self.sk)
        encoded_params_str = params_str.encode("utf-8")
        return hashlib.md5(encoded_params_str).hexdigest()

    def get_signature(self):
        timestamp = int(time.time())
        random_str = self.__random_password(20)
        sig = self.__signature(random_str, timestamp)
        params = {
            "timestamp": timestamp,
            "random": random_str,
            "app_id": self.ak,
            "sign": sig,
        }
        return params

    def embed(self, text):
        """
        Embed text using the server

        Args:
            text: The input text to embed

        Returns:
            requests.Response object
        """
        params = self.get_signature()
        params["body"] = text
        params["content_id"] = f"test_{int(time.time())}"

        headers = {"Content-Type": "application/json"}

        rsp = requests.post(self.url, data=json.dumps(params), timeout=self.timeout, headers=headers)
        result = json.loads(rsp.text)
        return result


class ConanWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        **kwargs,
    ) -> None:

        AK = os.getenv("CONAN_AK")
        SK = os.getenv("CONAN_SK")
        self.client = Client(ak=AK, sk=SK, url="https://ai.om.qq.com/api/conan/v2")
        self.embed_func = self.client.embed
        self.model_name = model_name


    def encode(
        self,
        sentences: list[str],
        **kwargs: Any,
    ) -> np.ndarray:

        embeddings = []
        
        for sentence in sentences:
            try:
                res = self.client.embed(sentence)['embedding']
            except Exception as e:
                # Sleep due to too many requests
                logger.info("Sleeping for 10 seconds due to error", e)
                import time
                time.sleep(10)
                res = self.client.embed(sentence)['embedding']

            embeddings.append(res)

        return np.array(embeddings)
        

Conan_embedding_v2 = ModelMeta(
    name="TencentBAC/Conan-embedding-v2",
    revision="a9975bd04f07e0fa8e9ec52b4385e19fa2c76b25",
    release_date="2025-4-10",
    languages=[
        "eng_Latn",
        "zho_Hans",
    ],  # supported languages not specified
    loader=partial(  # type: ignore
        ConanWrapper,
        model_name="Conan_embedding_v2",
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
    training_datasets={
        **E5_MISTRAL_TRAINING_DATA,
        **bge_full_data,
        **conan_zh_datasets,
    }
    public_training_code=None,
    public_training_data=None,
)
