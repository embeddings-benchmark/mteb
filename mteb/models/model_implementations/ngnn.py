# Copyright (c) 2026 steffen181
# SPDX-License-Identifier: MIT
"""Standalone NGNN inference using caller-funded OpenAI base embeddings.

Place the immutable ``model.npz`` beside this module, or pass its local path.
The postprocessor uses CPU PyTorch 2.11.0; its singleton normalization and
``torch.topk(sorted=False)`` behavior are part of the published model revision.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_ARTIFACT_PATH = Path(__file__).with_name("model.npz")
MODEL_NAME = "steffen-negabo/ngnn-general-encoder-v1"
MODEL_REVISION = "d6969c26400944d4f5200ebddfdc04a083fd7b75"
ARTIFACT_REVISION = "sparse_ngnn_v1_api_20260711_469679b8"
ARTIFACT_SHA256 = "e55ba67998039ba7cb4837798b0464e06e5939245d3275802cbbd6f0aa3fcb6a"
BASE_EMBEDDING_MODEL = "text-embedding-3-large"
INPUT_DIM = 3072
OUTPUT_DIM = 512
MAX_INPUT_TOKENS = 8191
MAX_REQUEST_INPUTS = 2048
MAX_REQUEST_TOKENS = 300000


class NgnnGeneralEncoderError(RuntimeError):
    """An invalid input, artifact, or provider response prevented inference."""


class FrozenNgnnArtifact:
    """The fixed public artifact, with a cached CPU ridge-system factorization."""

    input_dim = INPUT_DIM
    rank = OUTPUT_DIM
    model_revision = ARTIFACT_REVISION

    def __init__(self, dictionary: np.ndarray, channel_scale: np.ndarray) -> None:
        self._basis = torch.as_tensor(dictionary, dtype=torch.float64, device="cpu")
        self._channel_scale = torch.as_tensor(
            channel_scale.T, dtype=torch.float32, device="cpu"
        )
        system = self._basis.T @ self._basis
        # Preserve the source solver's regularizer and float64 arithmetic.
        system = system + (1e-6 + 1e-8) * torch.eye(  # noqa: PLR6104 -- preserve source arithmetic
            OUTPUT_DIM, dtype=torch.float64, device="cpu"
        )
        self._cholesky, info = torch.linalg.cholesky_ex(system)
        if int(info.item()) != 0:
            raise NgnnGeneralEncoderError("Frozen artifact ridge factorization failed")

    @classmethod
    def from_file(cls, path: str | Path) -> FrozenNgnnArtifact:
        try:
            data = Path(path).read_bytes()
        except OSError:
            raise NgnnGeneralEncoderError(
                "Could not read frozen NGNN artifact"
            ) from None
        if hashlib.sha256(data).hexdigest() != ARTIFACT_SHA256:
            raise NgnnGeneralEncoderError("Frozen NGNN artifact SHA-256 mismatch")
        # Load the same verified bytes; no pickle or remote code is involved.
        try:
            with np.load(BytesIO(data), allow_pickle=False) as payload:
                dictionary = np.asarray(payload["dictionary"], dtype=np.float32)
                channel_scale = np.asarray(
                    payload["code_channel_scale"], dtype=np.float32
                )
        except Exception:
            raise NgnnGeneralEncoderError(
                "Could not load frozen NGNN artifact"
            ) from None
        return cls(dictionary, channel_scale)

    @torch.inference_mode()
    def transform(self, base_embeddings: object) -> np.ndarray:
        try:
            rows = np.asarray(base_embeddings, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            raise NgnnGeneralEncoderError("Base embeddings must be numeric") from None
        if rows.ndim != 2 or rows.shape[1] != INPUT_DIM:
            raise NgnnGeneralEncoderError(
                f"Base embeddings must have shape (rows, {INPUT_DIM})"
            )
        if not np.isfinite(rows).all():
            raise NgnnGeneralEncoderError(
                "Base embeddings must contain only finite values"
            )
        if not len(rows):
            return np.empty((0, OUTPUT_DIM), dtype=np.float32)
        output = []
        for row in rows:
            target = torch.as_tensor(
                row.reshape(-1, 1), dtype=torch.float32, device="cpu"
            ).contiguous()
            rhs = self._basis.T @ target.to(dtype=torch.float64)
            dense = torch.cholesky_solve(rhs, self._cholesky).to(dtype=torch.float32)
            # Normalization is across the singleton sample axis. Do not batch
            # this solve or change the top-k tie rule for the current revision.
            rms = dense.square().mean(dim=1).sqrt().clamp_min(1e-6)
            normalized = dense / rms.unsqueeze(1)
            indices = (
                normalized.abs().topk(256, dim=0, largest=True, sorted=False).indices
            )
            sparse = torch.zeros_like(normalized)
            sparse.scatter_(0, indices, normalized.gather(0, indices))
            output.append((sparse / self._channel_scale).T.numpy()[0])
        result = np.vstack(output).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise NgnnGeneralEncoderError(
                "Frozen artifact transform produced non-finite values"
            )
        return result


class NgnnGeneralEncoder:
    """Encode arbitrary text with the caller's OpenAI client or API key.

    ``client`` is an optional OpenAI-compatible client. Without it, ``api_key``
    or the caller's ``OPENAI_API_KEY`` environment variable is required.
    Constructing this object makes no embedding request; ``encode`` does.
    Empty or whitespace-only inputs produce zero vectors without a request.
    Long texts use a cl100k_base prefix of at most 8191 tokens.
    """

    model_name = MODEL_NAME
    model_revision = MODEL_REVISION

    def __init__(
        self,
        artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
        *,
        api_key: str | None = None,
        client: Any | None = None,  # noqa: ANN401 -- accepts OpenAI-compatible offline clients
        provider_batch_size: int = 100,
    ) -> None:
        if type(provider_batch_size) is not int or provider_batch_size <= 0:
            raise NgnnGeneralEncoderError(
                "provider_batch_size must be a positive integer"
            )
        self.artifact = FrozenNgnnArtifact.from_file(artifact_path)
        if client is None:
            key = (
                api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
            )
            if not isinstance(key, str) or not key.strip():
                raise NgnnGeneralEncoderError(
                    "Set OPENAI_API_KEY or pass your own api_key or client"
                )
            try:
                from openai import OpenAI

                client = OpenAI(api_key=key, max_retries=2)
            except Exception:
                raise NgnnGeneralEncoderError(
                    "Could not initialize base embedding provider"
                ) from None
        self.client = client
        self.provider_batch_size = provider_batch_size

    def encode(
        self,
        inputs: Iterable[Any] | Mapping[str, Any] | str,
        *,
        task_metadata: object | None = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: object | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        del task_metadata, hf_split, hf_subset, prompt_type, kwargs
        texts = _extract_texts(inputs)
        if not texts:
            return np.empty((0, OUTPUT_DIM), dtype=np.float32)
        prepared = _prepare_texts(texts)
        output = np.zeros((len(texts), OUTPUT_DIM), dtype=np.float32)
        if not any(text is not None for text, _ in prepared):
            return output
        vectors: dict[str, list[float]] = {}
        for batch in _request_batches(prepared, self.provider_batch_size):
            try:
                response = self.client.embeddings.create(
                    model=BASE_EMBEDDING_MODEL, input=batch, encoding_format="float"
                )
            except Exception:
                raise NgnnGeneralEncoderError(
                    "Base embedding provider request failed"
                ) from None
            try:
                data = list(response.data)
                if len(data) != len(batch):
                    raise ValueError("response count")
                by_index = {}
                for item in data:
                    if type(item.index) is not int or not 0 <= item.index < len(batch):
                        raise ValueError("response index")
                    if item.index in by_index:
                        raise ValueError("duplicate response index")
                    by_index[item.index] = item
                for index, text in enumerate(batch):
                    vectors[_text_hash(text)] = [
                        float(x) for x in by_index[index].embedding
                    ]
            except Exception:
                raise NgnnGeneralEncoderError(
                    "Base embedding provider response is malformed"
                ) from None
        rows = []
        positions = []
        for position, (text, _) in enumerate(prepared):
            if text is None:
                continue
            vector = vectors[_text_hash(text)]
            row = np.asarray(vector, dtype=np.float64)
            if row.ndim != 1 or row.shape[0] != INPUT_DIM:
                raise NgnnGeneralEncoderError(
                    "Base embedding provider vector has the wrong dimension"
                )
            if not np.isfinite(row).all():
                raise NgnnGeneralEncoderError(
                    "Base embedding provider vector must contain only finite values"
                )
            # Preserve the original Python-float sum and explicit float32 rounding.
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isfinite(norm) or norm == 0.0:
                raise NgnnGeneralEncoderError(
                    "Base embedding provider vector norm must be finite and non-zero"
                )
            rows.append(
                [
                    struct.unpack("!f", struct.pack("!f", value / norm))[0]
                    for value in vector
                ]
            )
            positions.append(position)
        try:
            output[positions] = self.artifact.transform(
                np.asarray(rows, dtype=np.float32)
            )
        except Exception:
            raise NgnnGeneralEncoderError("Frozen artifact transform failed") from None
        return output

    @staticmethod
    def similarity(embeddings1: object, embeddings2: object) -> np.ndarray:
        return _similarity_rows(embeddings1) @ _similarity_rows(embeddings2).T

    @staticmethod
    def similarity_pairwise(embeddings1: object, embeddings2: object) -> np.ndarray:
        left, right = _similarity_rows(embeddings1), _similarity_rows(embeddings2)
        if left.shape[0] != right.shape[0]:
            raise NgnnGeneralEncoderError(
                "Pairwise similarity inputs must contain the same number of rows"
            )
        return np.sum(left * right, axis=1)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepare_texts(texts: list[str]) -> list[tuple[str | None, int]]:
    tokenizer = None
    prepared = []
    for original_text in texts:
        text = original_text
        if not text.strip():
            prepared.append((None, 0))
            continue
        if tokenizer is None:
            import tiktoken

            tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text, disallowed_special=())
        if len(tokens) > MAX_INPUT_TOKENS:
            prefix = tokens[:MAX_INPUT_TOKENS]
            while True:
                text = tokenizer.decode(prefix)
                # A partial Unicode character at the cutoff can change the
                # token count when the decoded string is encoded again.
                tokens = tokenizer.encode(text, disallowed_special=())
                if len(tokens) <= MAX_INPUT_TOKENS:
                    break
                prefix = prefix[:-1]
        prepared.append((text, len(tokens)) if text.strip() else (None, 0))
    return prepared


def _request_batches(
    prepared: list[tuple[str | None, int]], batch_size: int
) -> Iterable[list[str]]:
    batch = []
    token_count = 0
    limit = min(batch_size, MAX_REQUEST_INPUTS)
    for text, count in prepared:
        if text is None:
            continue
        if batch and (len(batch) >= limit or token_count + count > MAX_REQUEST_TOKENS):
            yield batch
            batch = []
            token_count = 0
        batch.append(text)
        token_count += count
    if batch:
        yield batch


def _extract_texts(inputs: Iterable[Any] | Mapping[str, Any] | str) -> list[str]:
    if isinstance(inputs, str):
        return [inputs]
    if isinstance(inputs, Mapping):
        if "text" not in inputs:
            raise NgnnGeneralEncoderError("Text input batch must contain a text field")
        return _text_values(inputs["text"])
    try:
        iterator = iter(inputs)
    except TypeError:
        raise NgnnGeneralEncoderError("Text inputs must be iterable") from None
    texts = []
    for item in iterator:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, Mapping) and "text" in item:
            texts.extend(_text_values(item["text"]))
        else:
            raise NgnnGeneralEncoderError(
                "Text inputs must contain strings or batches with a text field"
            )
    return texts


def _text_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    try:
        values = list(value)
    except TypeError:
        raise NgnnGeneralEncoderError("Text fields must contain strings") from None
    if any(not isinstance(item, str) for item in values):
        raise NgnnGeneralEncoderError("Text fields must contain strings")
    return values


def _similarity_rows(value: object) -> np.ndarray:
    try:
        rows = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise NgnnGeneralEncoderError(
            "Similarity inputs must be numeric arrays"
        ) from None
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != OUTPUT_DIM:
        raise NgnnGeneralEncoderError(
            f"Similarity inputs must have shape (rows, {OUTPUT_DIM})"
        )
    if not np.isfinite(rows).all():
        raise NgnnGeneralEncoderError(
            "Similarity inputs must contain only finite values"
        )
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms == 0.0, 1.0, norms)


# MTEB integration. The preceding inference source is copied without changes
# from the public ngnn_general_encoder.py; regenerate with the repository's
# scripts/build_ngnn_mteb_wrapper.py when that source changes.
import tempfile  # noqa: E402 -- preserve the verbatim standalone inference module above
from urllib.request import urlopen  # noqa: E402

from mteb.models.model_meta import ModelMeta, ScoringFunction  # noqa: E402

PUBLIC_ARTIFACT_COMMIT = "83773ff1ad83accc729c8d748d6991077842fbc0"
ARTIFACT_URL = (
    "https://raw.githubusercontent.com/steffen181/frozen-ngnn-api-modesl/"
    f"{PUBLIC_ARTIFACT_COMMIT}/model.npz"
)


def _resolve_artifact(
    artifact_path: str | Path | None, cache_dir: str | Path | None
) -> Path:
    if artifact_path is not None:
        return Path(artifact_path)
    directory = (
        Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "ngnn"
    )
    target = directory / f"{ARTIFACT_SHA256}.npz"
    if target.is_file():
        # The encoder verifies cached bytes, including explicitly supplied files.
        return target
    try:
        with urlopen(ARTIFACT_URL, timeout=60) as response:  # noqa: S310 -- fixed HTTPS URL, no caller-controlled scheme
            data = response.read(8 * 1024 * 1024 + 1)
    except Exception:
        raise NgnnGeneralEncoderError(
            "Could not download public NGNN artifact"
        ) from None
    if hashlib.sha256(data).hexdigest() != ARTIFACT_SHA256:
        raise NgnnGeneralEncoderError("Public NGNN artifact SHA-256 mismatch")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, suffix=".partial", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def load_ngnn_model(
    model_name: str = MODEL_NAME,
    revision: str | None = MODEL_REVISION,
    *,
    artifact_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    client: Any | None = None,  # noqa: ANN401 -- accepts OpenAI-compatible offline clients
    provider_batch_size: int = 100,
    device: str | None = None,
    **kwargs: Any,
) -> NgnnGeneralEncoder:
    """Load the public fixed model; supply credentials through OPENAI_API_KEY.

    Do not pass secrets to mteb.get_model: MTEB records loader kwargs in result
    metadata. Offline tools may inject a compatible client directly through
    this loader, which assigns clean canonical metadata.
    """
    if "api_key" in kwargs:
        raise ValueError("Use OPENAI_API_KEY; MTEB serializes model keyword arguments")
    if kwargs:
        raise ValueError("Unsupported NGNN model keyword arguments")
    if model_name != MODEL_NAME:
        raise ValueError("Unsupported NGNN model name")
    if revision not in {None, MODEL_REVISION}:
        raise ValueError("Unsupported NGNN model revision")
    if device not in {None, "cpu"}:
        raise ValueError("This NGNN revision uses CPU inference")
    if torch.__version__.split("+")[0] != "2.11.0":
        raise RuntimeError(
            "This NGNN revision requires torch==2.11.0 for top-k tie semantics"
        )
    model = NgnnGeneralEncoder(
        _resolve_artifact(artifact_path, cache_dir),
        client=client,
        provider_batch_size=provider_batch_size,
    )
    model.mteb_model_meta = ngnn_general_encoder.model_copy(deep=True)
    return model


ngnn_general_encoder = ModelMeta(
    name=MODEL_NAME,
    revision=MODEL_REVISION,
    loader=load_ngnn_model,
    release_date="2026-09-06",
    languages=["eng-Latn"],
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=8191,
    embed_dim=OUTPUT_DIM,
    license=None,  # MIT compressor; closed API base model has separate terms.
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    framework=["API", "PyTorch"],
    reference="https://github.com/steffen181/frozen-ngnn-api-modesl",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,  # Unknown training provenance of the OpenAI base model.
    adapted_from="openai/text-embedding-3-large",
    modalities=["text"],
    model_type=["dense"],
    contacts=["steffen181"],
    extra_requirements_groups=["openai"],
)
