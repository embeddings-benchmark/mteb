from __future__ import annotations

import base64
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

# embeddingPurpose values validated against the Bedrock API.
PURPOSE_DOCUMENT = "GENERIC_INDEX"
PURPOSE_QUERY = "GENERIC_RETRIEVAL"
PURPOSE_CLASSIFICATION = "CLASSIFICATION"
PURPOSE_CLUSTERING = "CLUSTERING"

_CLASSIFICATION_TASKS = {
    "Classification",
    "MultilabelClassification",
    "PairClassification",
}


class NovaMultimodalEmbeddingsModel(AbsEncoder):
    """Amazon Nova Multimodal Embeddings via the Bedrock synchronous InvokeModel API.

    Text and image only. Audio and video are supported by the model but are
    deferred: mteb decodes both before they reach the encoder (audio arrives as
    a float array, video as a torchcodec-decoded frame tensor), so sending them
    would mean re-encoding decoded media back into a container.
    """

    def __init__(
        self,
        model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0",
        region_name: str = "us-east-1",
        embedding_dimension: int = 3072,
        max_tokens: int = 8192,
        max_workers: int = 8,
        **kwargs: Any,
    ) -> None:
        import boto3
        from botocore.config import Config

        self._model_id = model_id
        self._embedding_dimension = embedding_dimension
        self._max_workers = max_workers
        # ~4.5 chars per token, the same heuristic BedrockModel uses for Titan
        self._max_chars = int(max_tokens * 4.5)

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
        )

    def _purpose(
        self, task_metadata: TaskMetadata, prompt_type: PromptType | None
    ) -> str:
        if task_metadata.type in _CLASSIFICATION_TASKS:
            return PURPOSE_CLASSIFICATION
        if task_metadata.type == "Clustering":
            return PURPOSE_CLUSTERING
        if prompt_type is not None and getattr(prompt_type, "value", prompt_type) == "query":
            return PURPOSE_QUERY
        return PURPOSE_DOCUMENT

    def _params(self, purpose: str) -> dict[str, Any]:
        return {
            "embeddingPurpose": purpose,
            "embeddingDimension": self._embedding_dimension,
        }

    def _text_payload(self, text: str, purpose: str) -> dict[str, Any]:
        params = self._params(purpose)
        params["text"] = {
            "truncationMode": "END",
            "value": text[: self._max_chars],
        }
        return {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": params}

    def _image_payload(self, image: Image.Image, purpose: str) -> dict[str, Any]:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
        params = self._params(purpose)
        params["image"] = {
            "format": "jpeg",
            "source": {"bytes": base64.b64encode(buffer.getvalue()).decode()},
        }
        return {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": params}

    def _invoke(self, payload: dict[str, Any]) -> list[float]:
        response = self._client.invoke_model(
            modelId=self._model_id, body=json.dumps(payload)
        )
        body = json.loads(response["body"].read())
        return body["embeddings"][0]["embedding"]

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
        purpose = self._purpose(task_metadata, prompt_type)
        show_progress_bar = kwargs.pop("show_progress_bar", False)

        embeddings: list[list[float]] = []
        pbar = tqdm(disable=not show_progress_bar, leave=False, desc="Nova MME")

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for batch in inputs:
                images = batch.get("image")
                texts = batch.get("text")

                if images and texts:
                    raise NotImplementedError(
                        "Interleaved text+image input is not yet supported for "
                        "Nova Multimodal Embeddings."
                    )
                if images:
                    payloads = [self._image_payload(im, purpose) for im in images]
                elif texts:
                    payloads = [self._text_payload(t, purpose) for t in texts]
                else:
                    raise ValueError(
                        f"Batch has no 'text' or 'image' key. Got: {list(batch.keys())}"
                    )

                embeddings.extend(pool.map(self._invoke, payloads))
                pbar.update(len(payloads))

        pbar.close()
        return np.array(embeddings, dtype=np.float32)


amazon_nova_2_multimodal_embeddings = ModelMeta(
    name="bedrock/amazon-nova-2-multimodal-embeddings",
    model_type=["dense"],
    revision="1",
    release_date="2025-10-28",
    languages=None,  # 200+ languages, not enumerated by the provider
    loader=NovaMultimodalEmbeddingsModel,
    loader_kwargs=dict(
        model_id="amazon.nova-2-multimodal-embeddings-v1:0",
        embedding_dimension=3072,
        max_tokens=8192,
    ),
    max_tokens=8192,
    embed_dim=3072,
    open_weights=False,
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    license=None,
    reference="https://docs.aws.amazon.com/nova/latest/userguide/nova-embeddings.html",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=False,
    modalities=["text", "image"],
    extra_requirements_groups=["boto3"],
)
