from __future__ import annotations

import datetime as dt
import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential
    from azure.storage.blob import BlobServiceClient, UserDelegationKey
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput


logger = logging.getLogger(__name__)


class MicrosoftEmbedding1Wrapper(AbsEncoder):
    """Azure Data AI multi-vector embeddings with inline results and MaxSim scoring.

    Set AZURE_DATA_AI_ENDPOINT or pass endpoint when loading the model.
    AZURE_DATA_AI_SCOPE optionally overrides the production token scope.
    Token vectors default to 256 dimensions; pass embed_dim=1024 for the full run.
    Images are temporarily uploaded under tmp/mteb-microsoft-embedding-1-large/
    in AZURE_DATA_AI_STAGING_CONTAINER in AZURE_DATA_AI_STAGING_ACCOUNT,
    accessed with a user delegation SAS, then deleted.
    The caller needs Embedding User on the inference account and Storage Blob
    Data Contributor on the staging account to upload, sign, and delete images.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        endpoint: str | None = None,
        scope: str | None = None,
        credential: TokenCredential | None = None,
        embed_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        from azure.data.ai import InferenceClient
        from azure.identity import DefaultAzureCredential

        if embed_dim not in {256, 1024}:
            raise ValueError(
                "Microsoft embedding multi-vector embed_dim must be 256 or 1024."
            )
        self.embed_dim = embed_dim
        endpoint = endpoint or os.environ.get("AZURE_DATA_AI_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Set AZURE_DATA_AI_ENDPOINT or pass endpoint to the model."
            )
        scope = scope or os.environ.get(
            "AZURE_DATA_AI_SCOPE", "https://dbinference.azure.com/.default"
        )
        self.model_name = model_name.rsplit("/", 1)[-1]
        self.revision = revision
        self._credential = credential or DefaultAzureCredential()
        self._owns_credential = credential is None
        self._client = InferenceClient(
            endpoint=endpoint,
            credential=self._credential,
            credential_scopes=[scope],
        )
        self._blob_service: BlobServiceClient | None = None
        self._delegation_key: UserDelegationKey | None = None
        self._delegation_key_expiry: dt.datetime | None = None

    def _embed(self, item: dict[str, str]) -> np.ndarray:
        result = self._client.get_embeddings(
            inputs=[item],
            model=self.model_name,
            model_version=self.revision,
            vector_kind="multi",
            dimensions=self.embed_dim,
            encoding_format="float",
            output_data_kind="float32",
            delivery="inline",
        )
        if result.model_version != self.revision:
            raise ValueError(
                f"Expected Microsoft embedding version {self.revision}, got {result.model_version}."
            )
        if len(result.data) != 1:
            raise ValueError("Expected one inline embedding per input.")
        item = result.data[0]
        embedding = np.asarray(item.multi_embedding, dtype=np.float32)
        if (
            embedding.ndim != 2
            or embedding.shape[0] == 0
            or embedding.shape[1] != self.embed_dim
            or tuple(item.shape) != embedding.shape
            or not np.isfinite(embedding).all()
        ):
            raise ValueError(f"Unexpected embedding shape: {embedding.shape}.")
        return embedding

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        from azure.core.exceptions import AzureError
        from azure.storage.blob import (
            BlobSasPermissions,
            BlobServiceClient,
            ContentSettings,
            generate_blob_sas,
        )

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

        name = f"tmp/mteb-microsoft-embedding-1-large/{uuid4().hex}.png"
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
        with BytesIO() as buffer:
            image.convert("RGB").save(buffer, format="PNG")
            blob.upload_blob(
                buffer.getvalue(),
                content_settings=ContentSettings(content_type="image/png"),
            )
        try:
            return self._embed({"kind": "image", "url": blob.url, "sasToken": sas})
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
    ) -> Array:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        embeddings = []
        for batch in tqdm(
            inputs, disable=not show_progress_bar, desc="Microsoft Embedding Encoding"
        ):
            if "image" in batch and "text" in batch:
                raise ValueError(
                    "Microsoft embeddings expect separate text and image inputs."
                )
            if "image" in batch:
                embeddings.extend(self._embed_image(image) for image in batch["image"])
            elif "text" in batch:
                kind = "query" if prompt_type == PromptType.query else "text"
                embeddings.extend(
                    self._embed({"kind": kind, "content": text})
                    for text in batch["text"]
                )
            else:
                raise ValueError("Microsoft embeddings expect a text or image batch.")

        if not embeddings:
            return np.empty((0, 0, self.embed_dim), dtype=np.float32)
        return pad_sequence(
            [torch.from_numpy(embedding) for embedding in embeddings], batch_first=True
        ).numpy()

    def close(self) -> None:
        self._client.close()
        if self._blob_service is not None:
            self._blob_service.close()
        if self._owns_credential:
            self._credential.close()


microsoft_embedding_1_large = ModelMeta(
    loader=MicrosoftEmbedding1Wrapper,
    loader_kwargs={"embed_dim": 256},
    name="Microsoft/microsoft-embedding-1-large",
    revision="1",
    release_date=None,
    languages=None,
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=[256, 1024],
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
)
