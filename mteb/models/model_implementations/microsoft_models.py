from __future__ import annotations

import datetime as dt
import logging
import math
import os
from contextlib import ExitStack, contextmanager
from io import BytesIO
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from collections.abc import Generator

    from azure.storage.blob import BlobServiceClient, UserDelegationKey
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import BatchedInput


logger = logging.getLogger(__name__)

# Limits enforced by the Microsoft Embedding 1 API
IMAGE_MAX_PIXELS = 50_000_000
MAX_IMAGE_BATCH_SIZE = 8
MAX_TEXT_BATCH_SIZE = 1


def _resize_image(image: Image.Image) -> Image.Image:
    from PIL import Image

    max_pixels = IMAGE_MAX_PIXELS
    width, height = image.size
    if width * height <= max_pixels:
        return image

    scale = math.sqrt(max_pixels / (width * height))
    size = (
        max(1, min(max_pixels, int(width * scale))),
        max(1, min(max_pixels, int(height * scale))),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


class MicrosoftEmbedding1Wrapper(AbsEncoder):
    """Multi-vector embeddings with inline results and MaxSim scoring.

    Suggested pre-flight checks:
    - Set AZURE_DATA_AI_ENDPOINT -> This is the main endpoint for inference client.
    - (Optional) Set AZURE_DATA_AI_SCOPE to override the production token scope.
    - (Image inputs) Set AZURE_DATA_AI_STAGING_ACCOUNT and
      AZURE_DATA_AI_STAGING_CONTAINER, and grant the caller Storage Blob Data
      Contributor on the staging account.

    Notes:
    - Images are temporarily uploaded under tmp/mteb-microsoft-embedding-1-large/
      in AZURE_DATA_AI_STAGING_CONTAINER in AZURE_DATA_AI_STAGING_ACCOUNT,
      accessed with a user delegation SAS, then deleted.
      The caller needs Embedding User on the inference account and Storage Blob
      Data Contributor on the staging account to upload, sign, and delete images.
    - The API supports both dense and multi-vector embeddings, but this wrapper
      explicitly uses only multi-vector embeddings.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        embed_dim: int = 1024,
        **kwargs: Any,
    ) -> None:
        from azure.data.ai import InferenceClient
        from azure.identity import DefaultAzureCredential

        if {"endpoint", "scope", "credential", "credentials"}.intersection(kwargs):
            raise ValueError(
                "Configure AZURE_DATA_AI_ENDPOINT, AZURE_DATA_AI_SCOPE, and "
                "DefaultAzureCredential outside model arguments to keep connection "
                "and authentication settings out of experiment metadata."
            )
        if embed_dim not in {128, 256, 512, 1024}:
            raise ValueError(
                "multi-vector embed_dim must be one of 128, 256, 512, or 1024."
            )
        self.embed_dim = embed_dim
        endpoint = os.environ.get("AZURE_DATA_AI_ENDPOINT")
        if not endpoint:
            raise ValueError("Set AZURE_DATA_AI_ENDPOINT to load the model.")
        scope = os.environ.get(
            "AZURE_DATA_AI_SCOPE", "https://dbinference.azure.com/.default"
        )
        self.model_name = model_name.rsplit("/", 1)[-1]
        self.revision = revision
        self._credential = DefaultAzureCredential()
        self._client = InferenceClient(
            endpoint=endpoint,
            credential=self._credential,
            credential_scopes=[scope],
        )
        self._blob_service: BlobServiceClient | None = None
        self._delegation_key: UserDelegationKey | None = None
        self._delegation_key_expiry: dt.datetime | None = None

    def _embed(self, inputs: list[dict[str, str]]) -> list[np.ndarray]:
        if not inputs:
            return []
        max_batch_size = (
            MAX_IMAGE_BATCH_SIZE
            if any(item["kind"] == "image" for item in inputs)
            else MAX_TEXT_BATCH_SIZE
        )
        if len(inputs) > max_batch_size:
            raise ValueError(
                f"API accepts at most {max_batch_size} inputs per request."
            )
        result = self._client.get_embeddings(
            inputs=inputs,
            model=self.model_name,
            vector_kind="multi",
            dimensions=self.embed_dim,
            encoding_format="float",
            output_data_kind="float16",
            delivery="inline",
        )
        if len(result.data) != len(inputs):
            raise ValueError("Expected one inline embedding per input.")
        embeddings = []
        for index, item in enumerate(sorted(result.data, key=lambda item: item.index)):
            if item.index != index:
                raise ValueError("Expected one inline embedding per input index.")
            embedding = np.asarray(item.multi_embedding, dtype=np.float16)
            if (
                embedding.ndim != 2
                or embedding.shape[0] == 0
                or embedding.shape[1] != self.embed_dim
                or tuple(item.shape) != embedding.shape
                or not np.isfinite(embedding).all()
            ):
                raise ValueError(f"Unexpected embedding shape: {embedding.shape}.")
            embeddings.append(embedding)
        return embeddings

    @contextmanager
    def _stage_image(self, image: Image.Image) -> Generator[dict[str, str], None, None]:
        from azure.core.exceptions import AzureError
        from azure.storage.blob import (
            BlobSasPermissions,
            BlobServiceClient,
            ContentSettings,
            generate_blob_sas,
        )

        # Current gateway does not support inline image inputs, so upload the image to blob storage.
        account = os.environ.get("AZURE_DATA_AI_STAGING_ACCOUNT")
        container = os.environ.get("AZURE_DATA_AI_STAGING_CONTAINER")
        if not account or not container:
            raise ValueError(
                "Set AZURE_DATA_AI_STAGING_ACCOUNT and AZURE_DATA_AI_STAGING_CONTAINER for image inputs."
            )
        if self._blob_service is None:
            self._blob_service = BlobServiceClient(
                f"https://{account}.blob.core.windows.net", credential=self._credential
            )

        # Reuse a delegation key until it is within 20 minutes of expiry.
        now = dt.datetime.now(dt.UTC)
        if (
            self._delegation_key_expiry is None
            or now + dt.timedelta(minutes=20) >= self._delegation_key_expiry
        ):
            expiry = now + dt.timedelta(hours=1)
            self._delegation_key = self._blob_service.get_user_delegation_key(
                now - dt.timedelta(minutes=5), expiry
            )
            self._delegation_key_expiry = expiry

        # Temporary blobs are deleted after inference, even if uploading or inference fails.
        name = f"tmp/mteb-microsoft-embedding-1-large/{uuid4().hex}.jpg"
        sas = generate_blob_sas(
            account_name=account,
            container_name=container,
            blob_name=name,
            user_delegation_key=self._delegation_key,
            permission=BlobSasPermissions(read=True),
            protocol="https",
            start=now - dt.timedelta(minutes=5),
            expiry=now + dt.timedelta(minutes=15),
        )
        blob = self._blob_service.get_blob_client(container=container, blob=name)
        try:
            with BytesIO() as buffer:
                _resize_image(image).convert("RGB").save(
                    buffer, format="JPEG", quality=95, subsampling=0, optimize=True
                )
                blob.upload_blob(
                    buffer.getvalue(),
                    content_settings=ContentSettings(content_type="image/jpeg"),
                )
            yield {"kind": "image", "url": blob.url, "sasToken": sas}
        finally:
            try:
                blob.delete_blob()
            except AzureError:
                logger.warning(
                    "Could not delete temporary Microsoft embedding blob %s", name
                )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
            if "image" in batch and "text" in batch:
                raise ValueError("API expects separate text and image inputs.")
            if "image" in batch:
                for start in range(0, len(batch["image"]), MAX_IMAGE_BATCH_SIZE):
                    with ExitStack() as stack:
                        image_inputs = [
                            stack.enter_context(self._stage_image(image))
                            for image in batch["image"][
                                start : start + MAX_IMAGE_BATCH_SIZE
                            ]
                        ]
                        embeddings.extend(self._embed(image_inputs))
            elif "text" in batch:
                kind = "query" if prompt_type == PromptType.query else "text"
                for start in range(0, len(batch["text"]), MAX_TEXT_BATCH_SIZE):
                    text_inputs = [
                        {"kind": kind, "content": text}
                        for text in batch["text"][start : start + MAX_TEXT_BATCH_SIZE]
                    ]
                    embeddings.extend(self._embed(text_inputs))
            else:
                raise ValueError("API expects a text or image batch.")

        return embeddings

    def close(self) -> None:
        self._client.close()
        if self._blob_service is not None:
            self._blob_service.close()
        self._credential.close()


microsoft_embedding_1_large = ModelMeta(
    loader=MicrosoftEmbedding1Wrapper,
    name="Microsoft/microsoft-embedding-1-large",
    revision="1",
    release_date=None,
    languages=[
        "eng-Latn",
        "spa-Latn",
        "fra-Latn",
        "por-Latn",
        "ita-Latn",
    ],
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=[128, 256, 512, 1024],
    license=None,
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    framework=["API"],
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=None,
    modalities=["image", "text"],
    model_type=["late-interaction"],
    extra_requirements_groups=["microsoft"],
    reference="https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/microsoft-embedding-1",
)
