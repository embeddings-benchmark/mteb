from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Any

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import OutputDType

if TYPE_CHECKING:
    import torch
    from transformers import BaseVideoProcessor, BatchFeature

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import PromptType

QWEN3_VL_EMBEDDING_CITATION = """@article{qwen3vlembedding,
  title={Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking},
  author={Li, Mingxin and Zhang, Yanzhao and Long, Dingkun and Chen Keqin and Song, Sibo and Bai, Shuai and Yang, Zhibo and Xie, Pengjun and Yang, An and Liu, Dayiheng and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2601.04720},
  year={2026}
}"""

IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
DEFAULT_INSTRUCTION = "Represent the user's input."
# total_pixels of a video in the official script: 10 frames of 768 tokens
# https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/main/scripts/qwen3_vl_embedding.py
MAX_TOTAL_PIXELS = 10 * 768 * IMAGE_FACTOR * IMAGE_FACTOR


def resize_video(video: torch.Tensor, total_pixels: int) -> torch.Tensor:
    """Resize sampled frames as qwen-vl-utils' fetch_video does for a frame list.

    Each frame is first fitted to the patch grid (fetch_image), then the clip to the budget.
    https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L403-L475
    """
    import torch
    from qwen_vl_utils.vision_process import (
        FRAME_FACTOR,
        IMAGE_MAX_TOKEN_NUM,
        IMAGE_MIN_TOKEN_NUM,
        VIDEO_MAX_TOKEN_NUM,
        VIDEO_MIN_TOKEN_NUM,
        ceil_by_factor,
        smart_resize,
    )
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize

    nframes = ceil_by_factor(len(video), FRAME_FACTOR)
    if nframes > len(video):
        video = torch.cat([video, video[-1:].expand(nframes - len(video), -1, -1, -1)])
    height, width = smart_resize(
        video.shape[2],
        video.shape[3],
        factor=IMAGE_FACTOR,
        min_pixels=IMAGE_MIN_TOKEN_NUM * IMAGE_FACTOR * IMAGE_FACTOR,
        max_pixels=IMAGE_MAX_TOKEN_NUM * IMAGE_FACTOR * IMAGE_FACTOR,
    )
    video = resize(
        video.cpu(),
        [height, width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    min_pixels = VIDEO_MIN_TOKEN_NUM * IMAGE_FACTOR * IMAGE_FACTOR
    max_pixels = max(
        min(
            VIDEO_MAX_TOKEN_NUM * IMAGE_FACTOR * IMAGE_FACTOR,
            total_pixels / nframes * FRAME_FACTOR,
        ),
        int(min_pixels * 1.05),
    )
    height, width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return resize(
        video.float(),
        [height, width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def load_video_processor(
    model_name: str, total_pixels: int, **kwargs: Any
) -> BaseVideoProcessor:
    """Qwen3-VL video processor that sizes frames with `resize_video`.

    The processor's own resize spends one pixel budget on the whole clip, with no
    per-frame cap and no upscaling of small frames, unlike the official scripts.
    """
    from transformers import Qwen3VLVideoProcessor

    class VideoProcessor(Qwen3VLVideoProcessor):
        def _preprocess(
            self, videos: list[torch.Tensor], **kwargs: Any
        ) -> BatchFeature:
            videos = [resize_video(video, total_pixels) for video in videos]
            return super()._preprocess(videos, **(kwargs | {"do_resize": False}))

    return VideoProcessor.from_pretrained(model_name, **kwargs)


class Qwen3VLEmbeddingWrapper(InstructSentenceTransformerModel):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        # fps=1, 64 frames: FPS=1 and MAX_FRAMES=64 in the official script (the
        # generic Qwen3-VL video processor default is fps=2)
        # https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/main/scripts/qwen3_vl_embedding.py
        fps: float | None = 1.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoImageProcessor

        processor_kwargs = kwargs.get("processor_kwargs", {})

        image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            **kwargs,
        )
        # VideoCollator already samples frames; skip inner sampling.
        video_processor = load_video_processor(
            model_name,
            MAX_TOTAL_PIXELS,
            revision=revision,
            trust_remote_code=True,
            do_sample_frames=False,
            **kwargs,
        )
        processor_kwargs |= {
            "video_processor": video_processor,
            "image_processor": image_processor,
        }

        kwargs["processor_kwargs"] = processor_kwargs

        super().__init__(
            model_name,
            revision=revision,
            device=device,
            fps=fps,
            max_frames=max_frames,
            num_frames=num_frames,
            # at least 4 frames: FPS_MIN_FRAMES of qwen-vl-utils, which the official
            # script samples with; the processor fails on a single frame
            min_frames=4,
            **kwargs,
        )

    def get_task_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        instruction = super().get_task_instruction(task_metadata, prompt_type)
        instruction = instruction or DEFAULT_INSTRUCTION
        instruction = instruction.strip()
        # Checks if the last character is not punctuation and appends "." then
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction += "."
        return instruction


qwen3_vl_embedding_2b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    name="Qwen/Qwen3-VL-Embedding-2B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
    release_date="2026-01-08",
    modalities=["image", "text", "video"],
    n_parameters=2_127_532_032,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=7629,
    embed_dim=2048,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
    output_dtypes=OutputDType.BF16,
)

qwen3_vl_embedding_8b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    name="Qwen/Qwen3-VL-Embedding-8B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="2c4565515e0f265c6511776e7193b22c0968ddc7",
    release_date="2026-01-08",
    modalities=["image", "text", "video"],
    n_parameters=8_144_793_840,
    n_embedding_parameters=622_329_856,
    memory_usage_mb=30518,
    embed_dim=4096,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation=QWEN3_VL_EMBEDDING_CITATION,
    extra_requirements_groups=["multimodal-sbert"],
    output_dtypes=OutputDType.BF16,
)


dna_vl_steer_2b = ModelMeta(
    loader=Qwen3VLEmbeddingWrapper,
    name="dnotitia/DNA-VL-STEER-2B",
    adapted_from="Qwen/Qwen3-VL-Embedding-2B",
    model_type=["dense"],
    # Calibrated on all 36 XM3600 languages (the base entry declares eng-Latn only).
    languages=[
        "eng-Latn",
        "fra-Latn",
        "deu-Latn",
        "spa-Latn",
        "zho-Hans",
        "kor-Hang",
        "ben-Beng",
        "fil-Latn",
        "hin-Deva",
        "swa-Latn",
        "tel-Telu",
        "jpn-Jpan",
        "rus-Cyrl",
        "ara-Arab",
        "vie-Latn",
        "tur-Latn",
        "tha-Thai",
        "ell-Grek",
        "fas-Arab",
        "fin-Latn",
        "ind-Latn",
        "heb-Hebr",
        "pol-Latn",
        "ukr-Cyrl",
        "nld-Latn",
        "ita-Latn",
        "por-Latn",
        "ces-Latn",
        "swe-Latn",
        "hun-Latn",
        "ron-Latn",
        "hrv-Latn",
        "dan-Latn",
        "nor-Latn",
        "mri-Latn",
        "quz-Latn",
    ],
    open_weights=True,
    revision="521a9cead4b31920579f80dfc9609a7969ddba86",
    release_date="2026-08-27",
    modalities=["image", "text", "video"],
    n_parameters=2_127_532_032,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=7629,
    embed_dim=2048,
    license="apache-2.0",
    max_tokens=32768,
    reference="https://huggingface.co/dnotitia/DNA-VL-STEER-2B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    # Calibration corpus is machine-translated CC12M captions (36 languages, ~9.5M rows).
    # It contains no MTEB task data, so there is no leaderboard contamination.
    public_training_data=None,
    training_datasets=None,
    extra_requirements_groups=["multimodal-sbert"],
    output_dtypes=OutputDType.BF16,
)
