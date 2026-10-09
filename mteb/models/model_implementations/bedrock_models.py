from __future__ import annotations

import base64
import io
import json
import logging
import re
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

from .cohere_models import (
    model_prompts as cohere_model_prompts,
)
from .cohere_models import (
    supported_languages as cohere_supported_languages,
)

if TYPE_CHECKING:
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput


logger = logging.getLogger(__name__)

# Rough characters-per-token ratio used to pre-truncate text before sending it
# to Bedrock.
# https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html
CHARS_PER_TOKEN = 4.5


def get_bedrock_runtime_client(
    region_name: str | None = None,
    config: Any = None,  # noqa: ANN401 -- botocore Config; boto3 is an optional, untyped dep
) -> Any:  # noqa: ANN401 -- botocore client; boto3 is an optional, untyped dep
    """Create a bedrock-runtime client.

    Defaults to the region of the active boto3 session when none is given.
    """
    import boto3

    if region_name is None:
        region_name = boto3.session.Session().region_name
    if config is None:
        return boto3.client("bedrock-runtime", region_name)
    return boto3.client("bedrock-runtime", region_name, config=config)


def read_response_body(response: Any) -> dict[str, Any]:  # noqa: ANN401 -- raw botocore response
    """Read and JSON-decode the streaming body of an InvokeModel response."""
    return json.loads(response.get("body").read())


class BedrockModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        model_id: str,
        provider: str,
        max_tokens: int,
        model_prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._client = get_bedrock_runtime_client()

        self._model_id = model_id
        self._provider = provider.lower()

        if self._provider == "cohere":
            self.model_prompts = self.validate_task_to_prompt_name(model_prompts)
            self._max_batch_size = 96
            self._max_sequence_length = max_tokens * 4
        else:
            self._max_tokens = max_tokens

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
        inputs = [text for batch in inputs for text in batch["text"]]
        show_progress_bar = (
            False
            if "show_progress_bar" not in kwargs
            else kwargs.pop("show_progress_bar")
        )
        if self._provider == "amazon":
            return self._encode_amazon(inputs, show_progress_bar)
        if self._provider == "cohere":
            prompt_name = self.get_prompt_name(task_metadata, prompt_type)
            cohere_task_type = self.model_prompts.get(prompt_name, "search_document")
            return self._encode_cohere(inputs, cohere_task_type, show_progress_bar)
        raise ValueError(
            f"Unknown provider '{self._provider}'. Must be 'amazon' or 'cohere'."
        )

    def _encode_amazon(
        self, sentences: list[str], show_progress_bar: bool = False
    ) -> Array:
        from botocore.exceptions import ValidationError

        all_embeddings = []
        max_sequence_length = int(self._max_tokens * CHARS_PER_TOKEN)

        for sentence in tqdm(sentences, leave=False, disable=not show_progress_bar):
            if len(sentence) > max_sequence_length:
                truncated_sentence = sentence[:max_sequence_length]
            else:
                truncated_sentence = sentence

            try:
                embedding = self._embed_amazon(truncated_sentence)
                all_embeddings.append(embedding)

            except ValidationError as e:
                error_str = str(e)
                pattern = r"request input token count:\s*(\d+)"
                match = re.search(pattern, error_str)
                if match:
                    num_tokens = int(match.group(1))

                    ratio = 0.9 * (self._max_tokens / num_tokens)
                    dynamic_cutoff = int(len(truncated_sentence) * ratio)

                    embedding = self._embed_amazon(truncated_sentence[:dynamic_cutoff])
                    all_embeddings.append(embedding)
                else:
                    raise e

        return np.array(all_embeddings)

    def _encode_cohere(
        self,
        sentences: list[str],
        cohere_task_type: str,
        show_progress_bar: bool = False,
    ) -> Array:
        batches = [
            sentences[i : i + self._max_batch_size]
            for i in range(0, len(sentences), self._max_batch_size)
        ]

        all_embeddings = []

        for batch in tqdm(batches, leave=False, disable=not show_progress_bar):
            response = self._client.invoke_model(
                body=json.dumps(
                    {
                        "texts": [sent[: self._max_sequence_length] for sent in batch],
                        "input_type": cohere_task_type,
                    }
                ),
                modelId=self._model_id,
                accept="*/*",
                contentType="application/json",
            )
            all_embeddings.extend(self._to_numpy(response))

        return np.array(all_embeddings)

    def _embed_amazon(self, sentence: str) -> Array:
        response = self._client.invoke_model(
            body=json.dumps({"inputText": sentence}),
            modelId=self._model_id,
            accept="application/json",
            contentType="application/json",
        )
        return self._to_numpy(response)

    def _to_numpy(self, embedding_response: Any) -> Array:  # noqa: ANN401 -- raw botocore response
        response = read_response_body(embedding_response)
        key = "embedding" if self._provider == "amazon" else "embeddings"
        return np.array(response[key])


amazon_titan_embed_text_v1 = ModelMeta(
    name="bedrock/amazon-titan-embed-text-v1",
    model_type=["dense"],
    revision="1",
    release_date="2023-09-27",
    languages=None,  # not specified
    loader=BedrockModel,
    loader_kwargs=dict(
        model_id="amazon.titan-embed-text-v1",
        provider="amazon",
        max_tokens=8192,
    ),
    max_tokens=8192,
    embed_dim=1536,
    open_weights=False,
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
    license=None,
    reference="https://aws.amazon.com/about-aws/whats-new/2023/09/amazon-titan-embeddings-generally-available/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=False,
    extra_requirements_groups=["boto3"],
)

amazon_titan_embed_text_v2 = ModelMeta(
    name="bedrock/amazon-titan-embed-text-v2",
    model_type=["dense"],
    revision="1",
    release_date="2024-04-30",
    languages=None,  # not specified
    loader=BedrockModel,
    loader_kwargs=dict(
        model_id="amazon.titan-embed-text-v2:0",
        provider="amazon",
        max_tokens=8192,
    ),
    max_tokens=8192,
    embed_dim=1024,
    open_weights=False,
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
    license=None,
    reference="https://aws.amazon.com/about-aws/whats-new/2024/04/amazon-titan-text-embeddings-v2-amazon-bedrock/",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=False,
    extra_requirements_groups=["boto3"],
)
# Note: For the original Cohere API implementation, refer to:
# https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/cohere_models.py
# This implementation uses the Amazon Bedrock endpoint for Cohere models.
cohere_embed_english_v3 = ModelMeta(
    loader=BedrockModel,
    loader_kwargs=dict(
        model_id="cohere.embed-english-v3",
        provider="cohere",
        max_tokens=512,
        model_prompts=cohere_model_prompts,
    ),
    name="bedrock/cohere-embed-english-v3",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=False,
    reference="https://cohere.com/blog/introducing-embed-v3",
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
    max_tokens=512,
    embed_dim=1024,
    license=None,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    extra_requirements_groups=["boto3"],
)

cohere_embed_multilingual_v3 = ModelMeta(
    loader=BedrockModel,
    loader_kwargs=dict(
        model_id="cohere.embed-multilingual-v3",
        provider="cohere",
        max_tokens=512,
        model_prompts=cohere_model_prompts,
    ),
    name="bedrock/cohere-embed-multilingual-v3",
    model_type=["dense"],
    languages=cohere_supported_languages,
    open_weights=False,
    reference="https://cohere.com/blog/introducing-embed-v3",
    revision="1",
    release_date="2023-11-02",
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    public_training_code=None,
    public_training_data=None,  # assumed
    training_datasets=None,
    max_tokens=512,
    embed_dim=1024,
    license=None,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    extra_requirements_groups=["boto3"],
)


# embeddingPurpose values accepted by Nova Multimodal Embeddings.
NOVA_PURPOSE_DOCUMENT = "GENERIC_INDEX"
NOVA_PURPOSE_QUERY = "GENERIC_RETRIEVAL"
NOVA_PURPOSE_CLASSIFICATION = "CLASSIFICATION"
NOVA_PURPOSE_CLUSTERING = "CLUSTERING"

# 30 s: Nova 2 accepts "8K tokens or 30s of video and 30s of audio"
# https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
NOVA_MAX_VIDEO_SECONDS = 30
NOVA_MAX_AUDIO_SECONDS = 30
NOVA_VIDEO_MODE = "AUDIO_VIDEO_COMBINED"
# Sample rates the AAC encoder accepts; others are resampled to 48 kHz.
_AAC_SAMPLE_RATES = {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000}

# "Max length: 8192 characters" for text.value, and inline sources are capped
# at "25 MB" after base64 encoding.
# https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
NOVA_MAX_TEXT_CHARS = 8192
NOVA_MAX_INLINE_BYTES = 25_000_000  # read conservatively as decimal MB

# Retrieval queries use the purpose matching the modality of the documents
# searched; mixed-modality corpora keep GENERIC_RETRIEVAL (schema docs).
_NOVA_QUERY_PURPOSE_BY_DOC_MODALITY = {
    "text": "TEXT_RETRIEVAL",
    "image": "IMAGE_RETRIEVAL",
    "audio": "AUDIO_RETRIEVAL",
    "video": "VIDEO_RETRIEVAL",
}


def _mux_audio_into_mp4(
    video_mp4: bytes, audio: dict[str, Any], max_seconds: float
) -> bytes:
    """Add ``audio``, cut to ``max_seconds``, to ``video_mp4`` as an AAC track."""
    import av

    array = np.asarray(audio["array"], dtype=np.float32)
    sampling_rate = int(audio["sampling_rate"])
    if array.ndim > 1:
        array = array.mean(axis=0)
    max_samples = int(min(max_seconds, NOVA_MAX_AUDIO_SECONDS) * sampling_rate)
    array = np.clip(array[:max_samples], -1.0, 1.0)
    if array.size == 0:
        return video_mp4
    rate = sampling_rate if sampling_rate in _AAC_SAMPLE_RATES else 48_000

    output = io.BytesIO()
    with (
        av.open(io.BytesIO(video_mp4)) as source,
        av.open(output, mode="w", format="mp4") as target,
    ):
        video_in = source.streams.video[0]
        video_out = target.add_stream_from_template(video_in)
        audio_out = target.add_stream("aac", rate=rate, layout="mono")
        for packet in source.demux(video_in):
            if packet.dts is None:  # flush packet
                continue
            packet.stream = video_out
            target.mux(packet)

        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(array[None, :]), format="flt", layout="mono"
        )
        frame.sample_rate = sampling_rate
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        for resampled in [*resampler.resample(frame), *resampler.resample(None)]:
            target.mux(audio_out.encode(resampled))
        target.mux(audio_out.encode(None))
    return output.getvalue()


class NovaMultimodalEmbeddingsModel(AbsEncoder):
    """Amazon Nova Multimodal Embeddings via the Bedrock synchronous InvokeModel API.

    Supports text, image, audio and video.

    Video arrives as a torchcodec ``VideoDecoder``, so frames are re-encoded to
    mp4 at the source frame rate before upload; Nova validates the container
    against the declared format and does not accept every source container.
    Segments are capped at ``NOVA_MAX_VIDEO_SECONDS``. For video+audio items
    the audio is muxed into the mp4 as its audio track, which Nova embeds
    together with the frames (``AUDIO_VIDEO_COMBINED``).
    """

    def __init__(
        self,
        model_name: str | None = None,
        revision: str | None = None,
        model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0",
        region_name: str = "us-east-1",
        embedding_dimension: int = 3072,
        max_tokens: int = 8192,
        max_workers: int = 8,
        **kwargs: Any,
    ) -> None:
        from botocore.config import Config

        self._model_id = model_id
        self._embedding_dimension = embedding_dimension
        self._max_workers = max_workers
        self._max_chars = min(int(max_tokens * CHARS_PER_TOKEN), NOVA_MAX_TEXT_CHARS)
        self._client = get_bedrock_runtime_client(
            region_name,
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
        )

    @staticmethod
    def _purpose(task_metadata: TaskMetadata, prompt_type: PromptType | None) -> str:
        task_type = task_metadata.simplified_task_type
        if task_type in {"classification", "pair-classification"}:
            return NOVA_PURPOSE_CLASSIFICATION
        if task_type == "clustering":
            return NOVA_PURPOSE_CLUSTERING
        if prompt_type == PromptType.query:
            doc_modalities = (
                task_metadata.get_modalities(PromptType.document)
                if task_type == "retrieval"
                else []
            )
            # video+audio documents are AUDIO_VIDEO_COMBINED video embeddings
            if set(doc_modalities) == {"audio", "video"}:
                doc_modalities = ["video"]
            if len(doc_modalities) == 1:
                return _NOVA_QUERY_PURPOSE_BY_DOC_MODALITY[doc_modalities[0]]
            return NOVA_PURPOSE_QUERY
        return NOVA_PURPOSE_DOCUMENT

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

    def _audio_payload(self, audio: dict[str, Any], purpose: str) -> dict[str, Any]:
        array = np.asarray(audio["array"], dtype=np.float32)
        sampling_rate = int(audio["sampling_rate"])
        if array.ndim > 1:
            array = array.mean(axis=0)
        max_samples = NOVA_MAX_AUDIO_SECONDS * sampling_rate
        if array.shape[-1] > max_samples:
            array = array[..., :max_samples]
        pcm = (np.clip(array, -1.0, 1.0) * 32767).astype("<i2")

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sampling_rate)
            wav.writeframes(pcm.tobytes())

        params = self._params(purpose)
        params["audio"] = {
            "format": "wav",
            "source": {"bytes": base64.b64encode(buffer.getvalue()).decode()},
        }
        return {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": params}

    def _video_payload(
        self,
        video: Any,  # noqa: ANN401 -- raw video payload handed to the Bedrock SDK
        purpose: str,
        audio: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import torch
        from torch.nn import functional as F
        from torchcodec.encoders import VideoEncoder

        metadata = video.metadata
        frame_rate = metadata.average_fps
        if not frame_rate:
            raise ValueError("Video has no frame rate metadata; cannot re-encode.")

        max_frames = int(frame_rate * NOVA_MAX_VIDEO_SECONDS)
        num_frames = metadata.num_frames or max_frames
        keep = min(num_frames, max_frames)

        # num_frames from the container header can overshoot what actually
        # decodes, so drop trailing frames until the read succeeds.
        frames = None
        while keep > 0:
            try:
                frames = video.get_frames_at(list(range(keep))).data
                break
            except RuntimeError as error:
                if "no more frames" not in str(error):
                    raise
                keep -= 1
        if frames is None or keep == 0:
            raise ValueError("Video has no decodable frames.")

        # h264 requires even dimensions.
        height, width = frames.shape[-2:]
        if height % 2 or width % 2:
            frames = frames[..., : height - (height % 2), : width - (width % 2)]

        # Pin the codec rather than leaving it to the local ffmpeg build, and
        # halve the resolution until the base64 payload fits the inline limit.
        while True:
            encoded = VideoEncoder(frames, frame_rate=frame_rate).to_tensor(
                format="mp4", codec="h264", pixel_format="yuv420p"
            )
            raw = encoded.numpy().tobytes()
            if audio is not None:
                raw = _mux_audio_into_mp4(raw, audio, keep / frame_rate)
            source = base64.b64encode(raw)
            height, width = frames.shape[-2:]
            if len(source) <= NOVA_MAX_INLINE_BYTES or min(height, width) < 64:
                break
            # resize a few frames at a time to bound memory on long 4K clips
            size = (height // 4 * 2, width // 4 * 2)
            frames = torch.cat(
                [
                    F.interpolate(chunk.float(), size=size, mode="area")
                    .round()
                    .to(frames.dtype)
                    for chunk in frames.split(16)
                ]
            )

        params = self._params(purpose)
        params["video"] = {
            "embeddingMode": NOVA_VIDEO_MODE,
            "format": "mp4",
            "source": {"bytes": source.decode()},
        }
        return {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": params}

    def _invoke(self, payload: dict[str, Any]) -> list[float]:
        response = self._client.invoke_model(
            modelId=self._model_id, body=json.dumps(payload)
        )
        return read_response_body(response)["embeddings"][0]["embedding"]

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> Array:
        purpose = self._purpose(task_metadata, prompt_type)

        embeddings: list[list[float]] = []
        pbar = tqdm(disable=not show_progress_bar, leave=False, desc="Nova MME")

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for batch in inputs:
                images = batch.get("image")
                audios = batch.get("audio")
                videos = batch.get("video")
                texts = batch.get("text")

                present = [
                    k for k in ("image", "audio", "video", "text") if batch.get(k)
                ]
                if set(present) == {"audio", "video"}:
                    payloads = [
                        self._video_payload(v, purpose, audio=a)
                        for v, a in zip(videos, audios, strict=True)
                    ]
                elif len(present) > 1:
                    raise NotImplementedError(
                        "Interleaved multi-modality input is not yet supported "
                        f"for Nova Multimodal Embeddings. Got: {present}"
                    )
                elif images:
                    payloads = [self._image_payload(im, purpose) for im in images]
                elif audios:
                    payloads = [self._audio_payload(a, purpose) for a in audios]
                elif videos:
                    payloads = [self._video_payload(v, purpose) for v in videos]
                elif texts:
                    payloads = [self._text_payload(t, purpose) for t in texts]
                else:
                    raise ValueError(
                        "Batch has no 'text', 'image', 'audio' or 'video' key. "
                        f"Got: {list(batch.keys())}"
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
    modalities=["text", "image", "audio", "video"],
    extra_requirements_groups=["boto3", "video", "av"],
)
