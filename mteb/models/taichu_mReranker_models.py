from __future__ import annotations

import base64
import logging
import os

from io import BytesIO
from typing import Any, List
from functools import partial

import numpy as np
import requests
import torch
from PIL import Image

from mteb.model_meta import ModelMeta

logger = logging.getLogger(__name__)


def pil_to_base64(image, format="jpeg"):
    buffer = BytesIO()
    image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    encoded_bytes = base64.b64encode(img_bytes)
    return encoded_bytes.decode("utf-8")


def taichu_multimodal_rerank(query=None, candidate=None):
    auth_token = os.getenv("TAICHU_AUTH_TOKEN")
    model_name = "ZiDongTaiChu__Taichu-mReranker-v1.0"
    api_url = "https://ai-szr.wair.ac.cn/api/v1/infer/11502/v2/models/rerank/generate"

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "candidate": candidate,
    }

    try:
        response = requests.post(url=api_url, headers=headers, json=payload, timeout=10)

        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error ({http_err.response.status_code}): {http_err}")
    except requests.exceptions.JSONDecodeError:
        logger.error("Error:The response is not in valid JSON format")
    except requests.exceptions.Timeout:
        logger.error("Error:Request timeout")
    except Exception as e:
        logger.error(f"Unknown error: {str(e)}")

    return None


class TaiChuReranker:
    def __init__(
        self,
        model_name_or_path="ZiDongTaiChu/Taichu-mReranker-v1.0",
        max_tokens: int = 32000,
        **kwargs,
    ):
        self._model_name = model_name_or_path

    def predict(self, text: str, image: List[Image.Image], **kwargs):
        """
        Perform t2vd style rerank
        Args:
            text: query text
            image: list of image which stands for visual document of candidates
        Returns:
            list of score of candidates
        """
        image_base64_list = [pil_to_base64(image) for image in image]

        query = [{"type": "text", "text": text}]
        candidate = [
            [{"type": "image", "image": image_base64}]
            for image_base64 in image_base64_list
        ]
        result_response = taichu_multimodal_rerank(query, candidate)
        score = result_response["score"]
        assert len(score) == len(image), (
            f"Expected {len(image)} scores, got {len(score)}"
        )
        return score


TRAINING_DATA = {
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "DocVQA": ["train"],
    "InfoVQA": ["train"],
    "TATDQA": ["train"],
    "arXivQA": ["train"],
    "WebQA": ["train"],
    "EVQA": ["train"],
    "docmatix-ir": ["train"],
    "vdr-multilingual-train": ["train"],
    "colpali_train_set": ["train"],  # as it contains PDFs
    "VisRAG-Ret-Train-Synthetic-data": ["train"],
    "VisRAG-Ret-Train-In-domain-data": ["train"],
}

Taichu_mReranker_v1_0 = ModelMeta(
    name="ZiDongTaiChu/Taichu-mReranker-v1.0",
    revision="1",
    release_date="2025-08-26",
    languages=[
        "eng-Latn",
        "zho-Hans",
    ],
    loader=partial(
        TaiChuReranker,
        model_name_or_path="ZiDongTaiChu/Taichu-mReranker-v1.0",
        max_tokens=32768,
    ),
    max_tokens=32768,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    embed_dim=None,
    license=None,
    reference=None,
    framework=["API"],
    training_datasets=TRAINING_DATA,
    public_training_code=None,
    public_training_data=None,
    is_cross_encoder=True,
    modalities=["text", "image"],
)
