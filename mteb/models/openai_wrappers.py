"""OpenAI-compatible API Wrappers for MTEB.

This module provides wrappers for embedding and reranking models served via
OpenAI-compatible APIs (such as vLLM servers, OpenAI, or other compatible servers).

Classes:
    OpenAIBaseWrapper: Base class with shared HTTP connection logic
    OpenAIAPIEncodeWrapper: Wrapper for embedding models using /v1/embeddings endpoint
    OpenAIAPIRerankWrapper: Wrapper for reranking models using /v1/rerank endpoint
    OpenAIAPITokenEmbedWrapper: Wrapper for ColBERT-style multi-vector (late interaction) retrieval models using the /pooling endpoint

Examples:
    Embeddings with vLLM:
        >>> from mteb.models import OpenAIAPIEncodeWrapper
        >>> wrapper = OpenAIAPIEncodeWrapper(
        ...     endpoint_url="http://localhost:8000",
        ...     model_name="BAAI/bge-small-en-v1.5"
        ... )

    Multimodal embeddings with vLLM:
        >>> wrapper = OpenAIAPIEncodeWrapper(
        ...     endpoint_url="http://localhost:8000",
        ...     model_name="Qwen/Qwen3-VL-Embedding-2B",
        ...     modalities=["text", "image"],
        ... )

    Reranking with vLLM:
        >>> from mteb.models import OpenAIAPIRerankWrapper
        >>> wrapper = OpenAIAPIRerankWrapper(
        ...     endpoint_url="http://localhost:8001",
        ...     model_name="BAAI/bge-reranker-v2-m3"
        ... )

    ColBERT-style multi-vector (late interaction) retrieval with vLLM:
        >>> from mteb.models import OpenAIAPITokenEmbedWrapper
        >>> wrapper = OpenAIAPITokenEmbedWrapper(
        ...     endpoint_url="http://localhost:8000",
        ...     model_name="BAAI/bge-m3",
        ... )

    With OpenAI API:
        >>> wrapper = OpenAIAPIEncodeWrapper(
        ...     endpoint_url="https://api.openai.com/v1",
        ...     model_name="text-embedding-3-small",
        ...     api_key="sk-..."
        ... )
"""

from __future__ import annotations

import base64
import heapq
import io
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import requests
import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import create_dataloader
from mteb.models.abs_encoder import AbsEncoder, get_prompt
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import (
        Array,
        BatchedInput,
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )
    from mteb.types._encoder_io import AudioInputItem
    from mteb.types._metadata import Modalities

    MultimodalItem = tuple[
        str | None, Image.Image | None, AudioInputItem | None, torch.Tensor | None
    ]
    """A single (text, image, audio, video) item collected from a batch; any
    of the last three may be None depending on the task's modalities."""

logger = logging.getLogger(__name__)


def _image_to_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    """Convert a PIL image to a base64-encoded `data:` URL.

    Used to embed image content directly in JSON payloads for OpenAI-compatible
    multimodal endpoints (e.g. vLLM), which accept `image_url` content parts
    with either an HTTP(S) URL or an inline `data:` URI.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def _audio_to_data_url(audio: AudioInputItem) -> str:
    """Convert a resampled audio item to a base64-encoded WAV `data:` URL.

    `audio` is expected to already be mono and resampled, as produced by
    `mteb.models.modality_collators.AudioCollator`. Uses `torchcodec`'s own
    encoder — the same package already required to decode video — so audio
    and video share one encoding library.
    """
    try:
        from torchcodec.encoders import AudioEncoder  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(
            "Sending audio content to a multimodal endpoint requires "
            "`torchcodec`. Install it with `pip install torchcodec` (or "
            "`pip install mteb[video]`)."
        ) from e

    samples = torch.as_tensor(audio["array"], dtype=torch.float32).clamp(-1.0, 1.0)
    samples = samples.unsqueeze(0)  # (num_samples,) -> (1, num_samples), mono

    encoded = AudioEncoder(samples, sample_rate=audio["sampling_rate"]).to_tensor(
        format="wav"
    )
    encoded_b64 = base64.b64encode(encoded.numpy().tobytes()).decode("utf-8")
    return f"data:audio/wav;base64,{encoded_b64}"


def _video_to_data_url(video: torch.Tensor, *, fps: float | None) -> str:
    """Convert decoded video frames to a base64-encoded MP4 `data:` URL.

    `video` is a decoded frame tensor as produced by
    `mteb.models.modality_collators.VideoCollator`/`FramesCollator`
    (`torchcodec`'s default `(T, C, H, W)` uint8 RGB layout). Re-encoding to
    an actual video container is necessary because vLLM's `video_url`
    content part expects a real video file (or a URL to one), not raw
    frames. Uses `torchcodec`'s own encoder — the same package already
    required to decode video — so no extra dependency is needed beyond the
    `video` extra.

    Args:
        video: Decoded frame tensor.
        fps: Frame rate to encode the output container at. This is the same
            `fps` used to *sample* frames from the source video (see
            `VideoCollator`); if `None` (e.g. fixed `num_frames` sampling
            was used instead), a default of 2.0 is used since there's no
            source frame rate to fall back to.
    """
    fps = fps or 2.0

    try:
        from torchcodec.encoders import VideoEncoder  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(
            "Sending video content to a multimodal endpoint requires "
            "`torchcodec`. Install it with `pip install torchcodec` (or "
            "`pip install mteb[video]`)."
        ) from e

    frames = video.detach().cpu()
    if frames.is_floating_point():
        frames = frames.clamp(0, 1).mul(255).round()
    frames = frames.to(torch.uint8)

    if frames.shape[1] == 1:
        # VideoEncoder requires exactly 3 (RGB) channels.
        frames = frames.repeat(1, 3, 1, 1)

    # h264's yuv420p pixel format uses 4:2:0 chroma subsampling, storing the
    # color planes at half resolution; an odd height/width has no whole-pixel
    # half, so the encoder rejects it. Pad by replicating the edge pixel
    # (not blank padding, so it doesn't distort the frame) to make both
    # dimensions even.
    height, width = frames.shape[-2], frames.shape[-1]
    pad_height, pad_width = height % 2, width % 2
    if pad_height or pad_width:
        frames = torch.nn.functional.pad(
            frames, (0, pad_width, 0, pad_height), mode="replicate"
        )

    encoded = VideoEncoder(frames, frame_rate=fps).to_tensor(format="mp4")
    encoded_b64 = base64.b64encode(encoded.numpy().tobytes()).decode("utf-8")
    return f"data:video/mp4;base64,{encoded_b64}"


def _build_content_parts(
    text: str | None,
    image: Image.Image | None,
    audio: AudioInputItem | None,
    video: torch.Tensor | None,
    *,
    fps: float | None,
) -> list[dict[str, Any]]:
    """Build a list of OpenAI-style content parts from a multimodal item.

    Used both for vLLM's `messages` field (Chat Embeddings/Pooling APIs) and
    for `{"content": [...]}` blocks (rerank/score API) — both accept the
    same `image_url`/`audio_url`/`video_url`/`text` content-part shapes.

    Args:
        text: Text content, if any.
        image: Image content, if any.
        audio: Audio content, if any.
        video: Video content, if any; re-encoded via `_video_to_data_url`
            using `fps` (see there for the fallback when `fps` is None).
        fps: Frame rate to re-encode `video` at, if present.
    """
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append(
            {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}}
        )
    if audio is not None:
        content.append(
            {"type": "audio_url", "audio_url": {"url": _audio_to_data_url(audio)}}
        )
    if video is not None:
        content.append(
            {
                "type": "video_url",
                "video_url": {"url": _video_to_data_url(video, fps=fps)},
            }
        )
    if text:
        content.append({"type": "text", "text": text})
    return content


def _collect_multimodal_items(
    inputs: DataLoader[BatchedInput],
) -> list[MultimodalItem]:
    """Collect (text, image, audio, video) tuples from a batch, preserving order.

    `inputs`' collate function must already resolve raw "audio"/"video"
    dataset columns into usable arrays/tensors (see
    `OpenAIBaseWrapper._configure_collate_fn`) before this is called.
    """
    items: list[MultimodalItem] = []
    for batch in inputs:
        batch_texts = batch.get("text")
        batch_images = batch.get("image")
        batch_audios = batch.get("audio")
        batch_videos = batch.get("video")
        n = next(
            len(values)
            for values in (batch_videos, batch_audios, batch_images, batch_texts)
            if values is not None
        )
        for i in range(n):
            items.append(
                (
                    batch_texts[i] if batch_texts is not None else None,
                    batch_images[i] if batch_images is not None else None,
                    batch_audios[i] if batch_audios is not None else None,
                    batch_videos[i] if batch_videos is not None else None,
                )
            )
    return items


class OpenAIBaseWrapper:
    """Base class for OpenAI-compatible API wrappers.

    Provides shared HTTP connection, retry logic, and error handling.

    Args:
        endpoint_url: URL of the OpenAI-compatible server
        model_name: Name of the model to use
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds (default: 300)
        max_retries: Maximum number of retries for failed requests (default: 3)
        verify_ssl: Whether to verify SSL certificates (default: True)
        modalities: Modalities supported by the served model; pass a subset
            (e.g. `["text"]`) if the served model is text-only, so
            `mteb.evaluate`'s model/task modality check reflects that.
        fps: Target frames per second for video downsampling (see
            `VideoCollator`). Only used when the task exposes a "video"
            column.
        max_frames: Safety cap on the number of frames sampled per video.
        num_frames: If set, sample exactly this many frames per video
            (fixed-sample mode) instead of FPS-based sampling.
        target_sampling_rate: Sampling rate (Hz) audio is resampled to
            before being sent to the server. Defaults to 16000 when the
            task exposes an "audio" (or "video" with audio) column.
        max_samples: Maximum number of audio samples to keep per item. If
            None, no truncation is applied.
    """

    mteb_model_meta: ModelMeta | None

    def __init__(  # noqa: PLR0913
        self,
        endpoint_url: str,
        model_name: str,
        api_key: str | None = None,
        *,
        timeout: int = 300,
        max_retries: int = 3,
        verify_ssl: bool = True,
        modalities: list[Modalities] | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        target_sampling_rate: int | None = None,
        max_samples: int | None = None,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples

        # Create model metadata for MTEB compatibility
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={
                "name": model_name,
                "modalities": modalities or ["text"],
            }
        )

    def _configure_collate_fn(self, inputs: DataLoader[BatchedInput]) -> None:
        """Attach VideoCollator/AudioCollator to `inputs` if applicable.

        Raw "video"/"audio" dataset columns need decoding/resampling into
        usable tensors/arrays before batching (`create_dataloader` does not
        do this by default); callers that plan to read `batch["video"]` or
        `batch["audio"]` must apply the appropriate collator first,
        mirroring `InstructSentenceTransformerModel.encode`. A no-op if
        `inputs` isn't a `DataLoader` wrapping a dataset with `features`
        (e.g. a plain list of pre-batched dicts).
        """
        features = getattr(getattr(inputs, "dataset", None), "features", None)
        if features is None:
            return
        has_video = "video" in features
        has_audio = "audio" in features

        if has_video:
            from mteb.models.modality_collators import VideoCollator

            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.target_sampling_rate or 16000,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
                max_samples=self.max_samples,
            )
        elif has_audio:
            from mteb.models.modality_collators import AudioCollator

            inputs.collate_fn = AudioCollator(
                target_sampling_rate=self.target_sampling_rate or 16000,
                max_samples=self.max_samples,
            )

    def _make_request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Make an HTTP POST request with retry logic.

        Args:
            endpoint: API endpoint path (e.g., "/v1/embeddings")
            payload: JSON payload to send

        Returns:
            JSON response from the server

        Raises:
            RuntimeError: If all retries fail
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.endpoint_url}{endpoint}",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )

                # 4xx means the request itself was rejected (e.g. a
                # malformed payload, or `messages` sent to a model with no
                # chat template) - retrying an identical request won't
                # help, so fail immediately with the server's error body,
                # which usually explains why.
                if 400 <= response.status_code < 500:
                    raise RuntimeError(
                        f"Server rejected request to {endpoint} "
                        f"({response.status_code}): {response.text[:2000]}"
                    )

                response.raise_for_status()
                json_response: dict[str, Any] = response.json()
                return json_response

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Request timeout (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying..."
                    )
                    continue
                raise
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}): "
                        f"{e}. Retrying..."
                    )
                    continue
                raise RuntimeError(f"Failed to get response from server: {e}") from e

        # This should never be reached due to the raise above, but mypy needs it
        raise RuntimeError("Failed to get response after all retries")

    def _verify_server(self) -> None:
        """Verify that the server is reachable and get model info."""
        try:
            response = requests.get(
                f"{self.endpoint_url}/v1/models",
                timeout=10,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            models = response.json()

            # Check if our model is available
            available_models = [m["id"] for m in models.get("data", [])]
            if self.model_name not in available_models:
                logger.warning(
                    f"Model '{self.model_name}' not found in server. "
                    f"Available models: {available_models}"
                )
                # Still allow initialization - model name might be alias
                return

            logger.info(f"Successfully connected to server. Model: {self.model_name}")

        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to server at {self.endpoint_url}: {e}"
            ) from e


class OpenAIAPIEncodeWrapper(OpenAIBaseWrapper, AbsEncoder):
    """OpenAI-compatible API wrapper for MTEB embedding benchmarks.

    This wrapper communicates with embedding models served via OpenAI-compatible
    HTTP APIs using the /v1/embeddings endpoint. When a batch contains image,
    audio, or video content, it switches to vLLM's Chat Embeddings API (a
    `messages` field on the same endpoint) to embed that content together with
    text, following https://docs.vllm.ai/en/latest/examples/pooling/embed/.
    This requires a vLLM server started with a multimodal pooling model and,
    for some models, a matching `--chat-template` (see vLLM's
    `vision_embedding_online.py` example for per-model server flags). Audio is
    sent as WAV, video as MP4 (re-encoded from decoded frames via
    `torchcodec`).

    `messages` is rendered through the model's chat template, so — like
    `OpenAIAPITokenEmbedWrapper` — it only works with chat-template-capable
    (typically VLM-based) models; non-chat text encoders reject it with a 400
    ("...default chat template is no longer allowed..."). By default
    (`use_chat_template=True`), all batches, including pure text, are sent
    through `messages`. Note that `messages` is a vLLM-only extension — the
    real OpenAI API and other OpenAI-compatible servers don't support it for
    embeddings at all — and non-chat text-encoder vLLM models (e.g.
    `BAAI/bge-small-en-v1.5` without a chat template) will reject it; set
    `use_chat_template=False` for those, which sends pure-text batches via
    the plain `input` field instead (image/audio/video content still
    requires `messages` and a chat-capable model regardless of this flag).

    Args:
        endpoint_url: URL of the OpenAI-compatible server
        model_name: Name of the model to use
        api_key: Optional API key for authentication
        prompt_dict: A dictionary mapping task names to prompt strings
        use_instructions: Whether to use instructions from the prompt_dict
        instruction_template: A template or callable to format instructions
        apply_instruction_to_documents: Whether to apply instructions to
            documents (passages). Default True.
        timeout: Request timeout in seconds (default: 300)
        max_retries: Maximum number of retries for failed requests (default: 3)
        verify_ssl: Whether to verify SSL certificates (default: True)
        max_length: Maximum sequence length for truncation. If None,
            auto-detected from model metadata.
        modalities: Modalities supported by the served model. Defaults to
            `["text", "image", "audio", "video"]`; pass a subset (e.g.
            `["text"]`) if the served model is text-only.
        use_chat_template: Whether to send text-only batches through the
            Chat Embeddings API (`messages`) like image/audio/video batches.
            Default True (see class docstring); set False for the real
            OpenAI API, other non-vLLM OpenAI-compatible servers, or vLLM
            models with no chat template.
        fps: Target frames per second for video downsampling (see
            `VideoCollator`).
        max_frames: Safety cap on the number of frames sampled per video.
        num_frames: If set, sample exactly this many frames per video
            (fixed-sample mode) instead of FPS-based sampling.
        target_sampling_rate: Sampling rate (Hz) audio is resampled to before
            being sent to the server. Defaults to 16000.
        max_samples: Maximum number of audio samples to keep per item.
    """

    def __init__(  # noqa: PLR0913
        self,
        endpoint_url: str,
        model_name: str,
        api_key: str | None = None,
        *,
        prompt_dict: dict[str, str] | None = None,
        use_instructions: bool = False,
        instruction_template: (
            str | Callable[[str, PromptType | None], str] | None
        ) = None,
        apply_instruction_to_documents: bool = True,
        timeout: int = 300,
        max_retries: int = 3,
        verify_ssl: bool = True,
        max_length: int | None = None,
        modalities: list[Modalities] | None = None,
        use_chat_template: bool = True,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        target_sampling_rate: int | None = None,
        max_samples: int | None = None,
    ):
        """Initialize the OpenAI API wrapper for embeddings."""
        # Initialize base class
        super().__init__(
            endpoint_url=endpoint_url,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
            modalities=modalities or ["text", "image", "audio", "video"],
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            target_sampling_rate=target_sampling_rate,
            max_samples=max_samples,
        )

        # Embedding-specific attributes
        self.prompts_dict = prompt_dict
        self.use_instructions = use_instructions
        self.instruction_template = instruction_template
        self.apply_instruction_to_passages = apply_instruction_to_documents
        self.max_length = max_length
        self.use_chat_template = use_chat_template

        if use_instructions and instruction_template is None:
            raise ValueError(
                "To use instructions, an instruction_template must be provided. "
                "For example, `Instruction: {instruction}`"
            )

        if (
            isinstance(instruction_template, str)
            and "{instruction}" not in instruction_template
        ):
            raise ValueError(
                "Instruction template must contain the string '{instruction}'."
            )

        # Verify server and detect max_length
        self._verify_server()
        self._detect_max_length()

    def _detect_max_length(self) -> None:
        """Auto-detect max_length from model metadata if not provided."""
        if self.max_length is not None:
            return

        try:
            response = requests.get(
                f"{self.endpoint_url}/v1/models",
                timeout=10,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            models = response.json()

            for model in models.get("data", []):
                if model["id"] != self.model_name:
                    continue
                # vLLM returns max_model_len in model metadata
                max_model_len = model.get("max_model_len")
                if max_model_len:
                    self.max_length = max_model_len
                    logger.info(
                        f"Auto-detected max_length={self.max_length} from model "
                        f"metadata"
                    )
                break
        except Exception as e:
            # If we can't detect max_length, that's fine - will use model default
            logger.debug(f"Could not auto-detect max_length: {e}")

    def _get_embeddings(self, texts: list[str]) -> Array:
        """Get embeddings from the server via OpenAI-compatible API.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }

        # Add truncation parameter if max_length is set
        # Note: vLLM supports truncate_prompt_tokens, OpenAI uses different params
        if self.max_length:
            payload["truncate_prompt_tokens"] = self.max_length

        result = self._make_request("/v1/embeddings", payload)

        # Extract embeddings, keyed by index since the server may return
        # them out of order.
        embeddings_by_index = {
            item["index"]: item["embedding"] for item in result["data"]
        }

        # Validate all embeddings were returned
        missing_indices = [i for i in range(len(texts)) if i not in embeddings_by_index]
        if missing_indices:
            raise RuntimeError(
                f"Incomplete embeddings from server: expected {len(texts)} "
                f"embeddings, got {len(texts) - len(missing_indices)}. "
                f"Missing indices: {missing_indices[:10]}"
            )

        # Convert to numpy array, restoring the original order
        embeddings = [embeddings_by_index[i] for i in range(len(texts))]
        return np.array(embeddings, dtype=np.float32)

    def _get_multimodal_embeddings(
        self, items: list[MultimodalItem], prompt: str
    ) -> Array:
        """Get embeddings for multimodal items via vLLM's Chat Embeddings API.

        Each item is sent as a single-turn chat message whose content is a
        list of `image_url`/`audio_url`/`video_url`/`text` parts, matching
        the request shape used in
        https://docs.vllm.ai/en/latest/examples/pooling/embed/ (e.g.
        `vision_embedding_online.py`).

        Args:
            items: List of (text, image, audio, video) tuples, at least one
                field of which is set per item.
            prompt: Instruction/prompt prefix to prepend to each item's text.

        Returns:
            Array of embeddings
        """
        messages_batch = []
        for text, image, audio, video in items:
            combined_text = prompt + text if text else prompt
            content = _build_content_parts(
                combined_text, image, audio, video, fps=self.fps
            )
            messages_batch.append([{"role": "user", "content": content}])

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages_batch,
            "encoding_format": "float",
        }

        if self.max_length:
            payload["truncate_prompt_tokens"] = self.max_length

        result = self._make_request("/v1/embeddings", payload)

        embeddings_by_index = {
            item["index"]: item["embedding"] for item in result["data"]
        }

        missing_indices = [i for i in range(len(items)) if i not in embeddings_by_index]
        if missing_indices:
            raise RuntimeError(
                f"Incomplete embeddings from server: expected {len(items)} "
                f"embeddings, got {len(items) - len(missing_indices)}. "
                f"Missing indices: {missing_indices[:10]}"
            )

        embeddings = [embeddings_by_index[i] for i in range(len(items))]
        return np.array(embeddings, dtype=np.float32)

    def _encode_multimodal(
        self,
        items: list[MultimodalItem],
        *,
        prompt: str,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Array:
        """Encode multimodal items in batches via the Chat Embeddings API."""
        if not items:
            return np.array([], dtype=np.float32).reshape(0, 0)

        all_embeddings = []
        for i in tqdm(
            range(0, len(items), batch_size),
            desc="Encoding multimodal batches",
            disable=not show_progress_bar,
        ):
            batch_items = items[i : i + batch_size]
            batch_embeddings = self._get_multimodal_embeddings(batch_items, prompt)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        """Encode the given sentences using the OpenAI-compatible API.

        Args:
            inputs: The sentences to encode
            task_metadata: The metadata of the task
            hf_split: Split of current task
            hf_subset: Subset of current task
            prompt_type: The type of prompt (query or passage)
            batch_size: Batch size for processing (default: 32)
            show_progress_bar: Whether to show progress bar (default: True)
            **kwargs: Additional arguments (precision, etc.)

        Returns:
            The encoded sentences as embeddings
        """
        # Determine prompt to use
        prompt = ""
        if self.use_instructions and self.prompts_dict is not None:
            prompt = self.get_task_instruction(task_metadata, prompt_type)
        elif self.prompts_dict is not None:
            prompt_name = self.get_prompt_name(task_metadata, prompt_type)
            if prompt_name is not None:
                prompt = self.prompts_dict.get(prompt_name, "")

        # Skip instruction for documents if configured
        if (
            self.use_instructions
            and self.apply_instruction_to_passages is False
            and prompt_type == PromptType.document
        ):
            logger.info(f"No instruction used, because prompt type = {prompt_type}")
            prompt = ""
        elif prompt:
            logger.info(
                f"Using instruction: '{prompt}' for task: '{task_metadata.name}' "
                f"prompt type: '{prompt_type}'"
            )

        # Resolve raw "video"/"audio" dataset columns, then collect
        # (text, image, audio, video) tuples from batches, preserving order.
        self._configure_collate_fn(inputs)
        items = _collect_multimodal_items(inputs)

        if self.use_chat_template or any(
            image is not None or audio is not None or video is not None
            for _, image, audio, video in items
        ):
            return self._encode_multimodal(
                items,
                prompt=prompt,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            )

        texts = [prompt + (text or "") for text, _, _, _ in items]

        # Handle empty input
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 0)

        # Process in batches to avoid overwhelming the server
        all_embeddings = []

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Encoding batches",
            disable=not show_progress_bar,
        ):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self._get_embeddings(batch_texts)
            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        return embeddings


class OpenAIAPIRerankWrapper(OpenAIBaseWrapper):
    """OpenAI-compatible API wrapper for MTEB reranking benchmarks.

    This wrapper communicates with reranking models served via OpenAI-compatible
    HTTP APIs using the /v1/rerank endpoint. Queries or documents that carry
    an image or video are sent as `{"content": [...]}` blocks containing
    `image_url`/`video_url`/`text` parts, matching the multimodal rerank
    request shape documented in
    https://docs.vllm.ai/en/latest/examples/pooling/score/ (e.g.
    `vision_rerank_api_online.py`). This requires a vLLM server started with a
    vision-language pooling/reranker model. Audio is *not* supported here:
    vLLM's rerank/score content-part schema (`ScoreContentPartParam`) has no
    audio variant, unlike the Chat Embeddings/Pooling APIs used by
    `OpenAIAPIEncodeWrapper`/`OpenAIAPITokenEmbedWrapper`.

    Args:
        endpoint_url: URL of the OpenAI-compatible server
        model_name: Name of the reranking model to use
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds (default: 300)
        max_retries: Maximum number of retries for failed requests (default: 3)
        verify_ssl: Whether to verify SSL certificates (default: True)
        modalities: Modalities supported by the served model. Defaults to
            `["text", "image", "video"]` (no audio; see above); pass a
            subset (e.g. `["text"]`) if the served model is text-only.
        fps: Target frames per second for video downsampling (see
            `VideoCollator`); also used as the encoded output frame rate.
        max_frames: Safety cap on the number of frames sampled per video.
        num_frames: If set, sample exactly this many frames per video
            (fixed-sample mode) instead of FPS-based sampling.
    """

    def __init__(  # noqa: PLR0913
        self,
        endpoint_url: str,
        model_name: str,
        api_key: str | None = None,
        *,
        timeout: int = 300,
        max_retries: int = 3,
        verify_ssl: bool = True,
        modalities: list[Modalities] | None = None,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
    ):
        """Initialize the OpenAI Rerank wrapper."""
        # Initialize base class
        super().__init__(
            endpoint_url=endpoint_url,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
            modalities=modalities or ["text", "image", "video"],
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
        )

        # Verify server is reachable
        self._verify_server()

    def _rerank(
        self,
        query: str | dict[str, Any],
        documents: list[str | dict[str, Any]],
        top_k: int | None = None,
    ) -> Array:
        """Get relevance scores for query-document pairs.

        Args:
            query: The query. Either a plain string, or a
                `{"content": [...]}` dict for multimodal (image and/or text)
                queries, built by `_to_score_input`.
            documents: List of documents to rank. Each document is either a
                plain string or a `{"content": [...]}` multimodal dict.
            top_k: Optional number of top results to return

        Returns:
            Array of relevance scores in the same order as input documents
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
        }

        if top_k is not None:
            payload["top_n"] = top_k

        result = self._make_request("/v1/rerank", payload)

        # Extract scores in original document order
        # The API returns documents sorted by relevance, so we need to reorder
        scores = [0.0] * len(documents)
        for item in result["results"]:
            original_index = item["index"]
            scores[original_index] = item["relevance_score"]

        return np.array(scores, dtype=np.float32)

    @staticmethod
    def _to_score_input(
        text: str | None,
        image: Image.Image | None,
        video: torch.Tensor | None,
        *,
        fps: float | None,
    ) -> str | dict[str, Any]:
        """Build a vLLM rerank/score input.

        Returns a plain string, or a multimodal `{"content": [...]}` dict
        when an image and/or video is present.
        """
        if image is None and video is None:
            return text or ""

        content = _build_content_parts(text, image, None, video, fps=fps)
        return {"content": content}

    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> Array:
        """Predict relevance scores for query-document pairs.

        Args:
            inputs1: Queries (first input)
            inputs2: Documents (second input)
            task_metadata: The metadata of the task
            hf_split: Split of current task
            hf_subset: Subset of current task
            prompt_type: The type of prompt
            batch_size: Batch size for processing (default: 32)
            show_progress_bar: Whether to show progress bar (default: True)
            top_k: Optional number of top results to return per query
            **kwargs: Additional arguments

        Returns:
            Relevance scores for each query-document pair
        """
        # Collect all queries and documents (text and, optionally,
        # images/video); resolve raw "video" columns first.
        self._configure_collate_fn(inputs1)
        self._configure_collate_fn(inputs2)
        query_items = _collect_multimodal_items(inputs1)
        document_items = _collect_multimodal_items(inputs2)

        # Handle empty input
        if not query_items and not document_items:
            return np.array([], dtype=np.float32)

        # Expect equal-length queries and documents
        if len(query_items) != len(document_items):
            raise ValueError(
                f"Expected equal number of queries and documents, got "
                f"{len(query_items)} queries and {len(document_items)} documents"
            )

        # Pairwise scoring - one query per document
        all_scores = []
        for i in tqdm(
            range(0, len(query_items), batch_size),
            desc="Reranking batches",
            disable=not show_progress_bar,
        ):
            batch_queries = query_items[i : i + batch_size]
            batch_docs = document_items[i : i + batch_size]

            # Score each pair individually
            batch_scores = []
            for (query_text, query_image, _, query_video), (
                doc_text,
                doc_image,
                _,
                doc_video,
            ) in zip(batch_queries, batch_docs):
                query = self._to_score_input(
                    query_text, query_image, query_video, fps=self.fps
                )
                document = self._to_score_input(
                    doc_text, doc_image, doc_video, fps=self.fps
                )
                score = self._rerank(query, [document], top_k)[0]
                batch_scores.append(score)

            all_scores.extend(batch_scores)

        scores = np.array(all_scores, dtype=np.float32)
        return scores


def _max_sim_scores(
    query_embedding: NDArray[np.floating], doc_embeddings: list[NDArray[np.floating]]
) -> list[float]:
    """Compute ColBERT-style MaxSim (late interaction) scores.

    For each document, computes the token-level similarity matrix between
    the query and document embeddings, takes the max over document tokens
    for each query token, then sums over query tokens.

    Args:
        query_embedding: Query embedding of shape `(num_query_tokens, dim)`.
        doc_embeddings: List of document embeddings, each of shape
            `(num_doc_tokens, dim)`.

    Returns:
        One MaxSim score per document, in the same order as `doc_embeddings`.
    """
    scores = []
    for doc_embedding in doc_embeddings:
        similarity = query_embedding @ doc_embedding.T
        scores.append(float(similarity.max(axis=-1).sum()))
    return scores


class OpenAIAPITokenEmbedWrapper(OpenAIBaseWrapper):
    """OpenAI-compatible API wrapper for ColBERT-style multi-vector retrieval models.

    Served via vLLM's Pooling API using late (token) interaction.

    Unlike `OpenAIAPIEncodeWrapper`, which returns a single fixed-size vector
    per input, this wrapper requests per-token embeddings (shape
    `(num_tokens, dim)`) from vLLM's `/pooling` endpoint, following
    https://docs.vllm.ai/en/latest/examples/pooling/token_embed/. It
    implements `SearchProtocol` directly: `index()` encodes and keeps the
    corpus' multi-vector embeddings in memory, and `search()` scores queries
    against them (or against `top_ranked` candidates, for reranking tasks)
    via brute-force MaxSim (late interaction) — no ANN index or extra
    dependency (e.g. PyLate) is used, so this scales linearly with corpus
    size rather than using an approximate index.

    The server must be started with a pooling model whose pooler task is
    `token_embed`, e.g. for a text-only ColBERT model:

        vllm serve BAAI/bge-m3 --pooler-config.task token_embed

    or, for a multimodal (image + text) late interaction model:

        vllm serve TomoroAI/tomoro-colqwen3-embed-4b --max-model-len 4096

    Image, audio, and video items are sent one request at a time via the Chat
    Pooling API (a `messages` field on `/pooling`), following
    `colqwen3_token_embed_online.py`; unlike `/v1/embeddings`, vLLM's
    `/pooling` endpoint does not support batching multiple chat conversations
    into a single request. `messages` is rendered through the model's chat
    template, so it only works with chat-template-capable (typically
    VLM-based) pooling models — non-chat text encoders like `BAAI/bge-m3`
    reject it with a 400 ("...default chat template is no longer
    allowed..."). By default (`use_chat_template=True`) all items, including
    pure text, are sent through `messages`; set `use_chat_template=False` for
    text-only models without a chat template, which instead batches text via
    the plain `input` field (image/audio/video items still require
    `messages` and a chat-capable model regardless of this flag). Audio is
    sent as WAV, video as MP4 (re-encoded from decoded frames; requires the
    `av` package).

    Args:
        endpoint_url: URL of the OpenAI-compatible server
        model_name: Name of the model to use
        api_key: Optional API key for authentication
        prompt_dict: A dictionary mapping task names to prompt strings
        timeout: Request timeout in seconds (default: 300)
        max_retries: Maximum number of retries for failed requests (default: 3)
        verify_ssl: Whether to verify SSL certificates (default: True)
        modalities: Modalities supported by the served model. Defaults to
            `["text", "image", "audio", "video"]`; pass a subset (e.g.
            `["text"]`) if the served model is text-only.
        use_chat_template: Whether to send text-only items through the Chat
            Pooling API (`messages`) like image/audio/video items. Default
            True. Set False for text-only pooling models that don't define a
            chat template (e.g. `BAAI/bge-m3`), so text is instead batched
            via the plain `input` field.
        fps: Target frames per second for video downsampling (see
            `VideoCollator`).
        max_frames: Safety cap on the number of frames sampled per video.
        num_frames: If set, sample exactly this many frames per video
            (fixed-sample mode) instead of FPS-based sampling.
        target_sampling_rate: Sampling rate (Hz) audio is resampled to before
            being sent to the server. Defaults to 16000.
        max_samples: Maximum number of audio samples to keep per item.
    """

    def __init__(  # noqa: PLR0913
        self,
        endpoint_url: str,
        model_name: str,
        api_key: str | None = None,
        *,
        prompt_dict: dict[str, str] | None = None,
        timeout: int = 300,
        max_retries: int = 3,
        verify_ssl: bool = True,
        modalities: list[Modalities] | None = None,
        use_chat_template: bool = True,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        target_sampling_rate: int | None = None,
        max_samples: int | None = None,
    ):
        """Initialize the OpenAI API wrapper for token-level (ColBERT-style) embeddings."""
        super().__init__(
            endpoint_url=endpoint_url,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
            modalities=modalities or ["text", "image", "audio", "video"],
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            target_sampling_rate=target_sampling_rate,
            max_samples=max_samples,
        )
        self.mteb_model_meta = self.mteb_model_meta.model_copy(  # type: ignore[union-attr]
            update={"similarity_fn_name": ScoringFunction.MAX_SIM}
        )

        self.prompt_dict = prompt_dict
        self.use_chat_template = use_chat_template
        self._corpus_ids: list[str] | None = None
        self._corpus_embeddings: list[NDArray[np.floating]] | None = None

        # Verify server is reachable
        self._verify_server()

    def _pooling_request(self, payload: dict[str, Any]) -> list[NDArray[np.floating]]:
        """Send a request to vLLM's /pooling endpoint.

        Returns per-item multi-vector (token-level) embeddings.
        """
        result = self._make_request("/pooling", payload)

        embeddings_by_index = {
            item["index"]: np.array(item["data"], dtype=np.float32)
            for item in result["data"]
        }

        expected = len(result["data"])
        missing_indices = [i for i in range(expected) if i not in embeddings_by_index]
        if missing_indices:
            raise RuntimeError(
                f"Incomplete pooling output from server: expected "
                f"{expected} items, missing indices: {missing_indices[:10]}"
            )

        return [embeddings_by_index[i] for i in range(expected)]

    def _encode_texts(self, texts: list[str]) -> list[NDArray[np.floating]]:
        """Get multi-vector embeddings for a batch of texts via /pooling.

        Uses the plain `input` field (`PoolingCompletionRequest`), which
        works for any pooling model. Pure-text items must go through this
        path rather than `_encode_item`'s Chat Pooling API: `messages` is
        rendered through the model's chat template, and most text-only
        pooling models (e.g. BAAI/bge-m3) don't define one, so vLLM rejects
        it with a 400 ("...default chat template is no longer allowed...").
        """
        payload: dict[str, Any] = {"model": self.model_name, "input": texts}
        return self._pooling_request(payload)

    def _encode_item(
        self,
        text: str | None,
        image: Image.Image | None,
        audio: AudioInputItem | None,
        video: torch.Tensor | None,
    ) -> NDArray[np.floating]:
        """Get a multi-vector embedding for a single image/audio/video item.

        Uses the Chat Pooling API (`messages`), required for non-text
        content and only supported by chat-template-capable (typically
        VLM-based) pooling models. vLLM's `/pooling` endpoint only accepts a
        single conversation per request, so these items cannot be batched.
        """
        content = _build_content_parts(text, image, audio, video, fps=self.fps)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
        }
        return self._pooling_request(payload)[0]

    def _encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> list[NDArray[np.floating]]:
        """Encode inputs as multi-vector (per-token) embeddings.

        Args:
            inputs: The sentences/images/audio/video to encode
            task_metadata: The metadata of the task
            hf_split: Split of current task
            hf_subset: Subset of current task
            prompt_type: The type of prompt (query or document)
            batch_size: Batch size used for batching pure-text requests when
                `use_chat_template=False` (default: 32). Image/audio/video
                items, and all items when `use_chat_template=True`, are
                always sent one request at a time (see class docstring).
            show_progress_bar: Whether to show progress bar (default: True)
            **kwargs: Additional arguments (unused)

        Returns:
            A list of per-item multi-vector embeddings, each of shape
            `(num_tokens, dim)`.
        """
        prompt = get_prompt(self.prompt_dict, task_metadata, prompt_type) or ""
        if prompt:
            logger.info(
                f"Using prompt: '{prompt}' for task: '{task_metadata.name}' "
                f"prompt type: '{prompt_type}'"
            )

        self._configure_collate_fn(inputs)
        items = _collect_multimodal_items(inputs)

        if self.use_chat_template:
            embeddings = []
            for text, image, audio, video in tqdm(
                items, desc="Pooling items", disable=not show_progress_bar
            ):
                combined_text = (prompt + text) if text else prompt
                embeddings.append(
                    self._encode_item(combined_text or None, image, audio, video)
                )
            return embeddings

        # use_chat_template=False: batch pure-text items via the plain
        # `input` field; image/audio/video items still need `messages`.
        text_indices = [
            i
            for i, (_, image, audio, video) in enumerate(items)
            if image is None and audio is None and video is None
        ]
        other_indices = [i for i in range(len(items)) if i not in set(text_indices)]

        embeddings_by_index: dict[int, NDArray[np.floating]] = {}

        for start in tqdm(
            range(0, len(text_indices), batch_size),
            desc="Pooling text batches",
            disable=not show_progress_bar,
        ):
            batch_idx = text_indices[start : start + batch_size]
            batch_texts = [prompt + (items[i][0] or "") for i in batch_idx]
            batch_embeddings = self._encode_texts(batch_texts)
            for idx, embedding in zip(batch_idx, batch_embeddings):
                embeddings_by_index[idx] = embedding

        for i in tqdm(
            other_indices,
            desc="Pooling multimodal items",
            disable=not show_progress_bar,
        ):
            text, image, audio, video = items[i]
            combined_text = (prompt + text) if text else prompt
            embeddings_by_index[i] = self._encode_item(
                combined_text or None, image, audio, video
            )

        return [embeddings_by_index[i] for i in range(len(items))]

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None,
    ) -> None:
        """Encode the corpus into multi-vector embeddings and keep them in memory.

        Args:
            corpus: Corpus dataset to index.
            task_metadata: Metadata of the task.
            hf_split: Split of current task.
            hf_subset: Subset of current task.
            encode_kwargs: Additional arguments to pass to `_encode`.
            num_proc: Number of processes to use for dataloading.
        """
        documents_loader = create_dataloader(
            corpus,
            task_metadata=task_metadata,
            prompt_type=PromptType.document,
            batch_size=encode_kwargs.get("batch_size", 32),
            num_proc=num_proc,
        )
        self._corpus_ids = [str(doc_id) for doc_id in corpus["id"]]
        self._corpus_embeddings = self._encode(
            documents_loader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.document,
            **encode_kwargs,
        )

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None,
    ) -> RetrievalOutputType:
        """Score queries against the indexed corpus using brute-force MaxSim.

        Args:
            queries: Queries to search with.
            task_metadata: Metadata of the task.
            hf_split: Split of current task.
            hf_subset: Subset of current task.
            top_k: Number of top documents to return per query.
            encode_kwargs: Additional arguments to pass to `_encode`.
            top_ranked: If given (reranking tasks), restricts scoring to
                these candidate document IDs per query instead of the full
                indexed corpus.
            num_proc: Number of processes to use for dataloading.

        Returns:
            Mapping of query ID to a mapping of document ID to relevance score.
        """
        if self._corpus_ids is None or self._corpus_embeddings is None:
            raise ValueError("Index is not built. Call index() before search().")

        queries_loader = create_dataloader(
            queries,
            task_metadata=task_metadata,
            prompt_type=PromptType.query,
            batch_size=encode_kwargs.get("batch_size", 32),
            num_proc=num_proc,
        )
        query_embeddings = self._encode(
            queries_loader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.query,
            **encode_kwargs,
        )
        query_ids = [row["id"] for row in queries]

        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self._corpus_ids)}

        results: RetrievalOutputType = {}
        for query_id, query_embedding in zip(query_ids, query_embeddings):
            if top_ranked is not None:
                candidate_ids = [
                    doc_id
                    for doc_id in top_ranked.get(query_id, [])
                    if doc_id in doc_id_to_idx
                ]
            else:
                candidate_ids = self._corpus_ids

            candidate_embeddings = [
                self._corpus_embeddings[doc_id_to_idx[doc_id]]
                for doc_id in candidate_ids
            ]
            scores = _max_sim_scores(query_embedding, candidate_embeddings)

            top_items = heapq.nlargest(
                top_k, zip(candidate_ids, scores), key=lambda item: item[1]
            )
            results[query_id] = dict(top_items)

        return results
