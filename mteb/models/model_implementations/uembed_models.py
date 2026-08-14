"""MTEB model definitions and adapter for Alibaba-NLP UEmbed.

The preprocessing and pooling logic is adapted from
https://github.com/Alibaba-NLP/UEmbed/blob/main/src/models/qwen35_embedding.py
(CC-BY-4.0). The checkpoint itself loads through native Transformers ``Auto``
classes. This adapter limits the public interface to ``last.normal`` and
``splade.last``, adds MTEB batching, and keeps sparse outputs in COO format.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import VideoCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

Pooling = Literal["last.normal", "splade.last"]
SUPPORTED_POOLING: tuple[Pooling, ...] = ("last.normal", "splade.last")

DEFAULT_INSTRUCTION = "Represent the user's input."
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
DEFAULT_FPS = 1.0
DEFAULT_MAX_FRAMES = 64
SPARSE_DIM = 184_016

_HUB_DOWNLOAD_KWARGS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "token",
)


def _hub_download_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: kwargs[key] for key in _HUB_DOWNLOAD_KWARGS if key in kwargs}


def _video_tensor_to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    """Convert MTEB's decoded ``[T, C, H, W]`` frames to UEmbed input."""
    from PIL import Image

    frames = video.detach().cpu()
    if frames.ndim != 4:
        raise ValueError(
            "Expected video frames with shape [T, C, H, W] or [T, H, W, C], "
            f"got {tuple(frames.shape)}"
        )

    if frames.shape[1] in {1, 3, 4}:
        frames = frames.permute(0, 2, 3, 1)
    elif frames.shape[-1] not in {1, 3, 4}:
        raise ValueError(
            f"Could not identify video channel axis in {tuple(frames.shape)}"
        )

    if torch.is_floating_point(frames):
        if frames.numel() and frames.max() <= 1:
            frames *= 255
        frames = frames.clamp(0, 255)
    frames = frames.to(torch.uint8).numpy()

    pil_frames = []
    for frame in frames:
        rgb_frame = frame[..., 0] if frame.shape[-1] == 1 else frame
        pil_frames.append(Image.fromarray(rgb_frame))
    return pil_frames


def _concatenate_sparse_batches(
    batches: list[torch.Tensor], sparse_dim: int
) -> torch.Tensor:
    if not batches:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
            size=(0, sparse_dim),
        ).coalesce()

    indices = []
    values = []
    row_offset = 0
    for batch in batches:
        coalesced = batch.coalesce()
        batch_indices = coalesced.indices().clone()
        batch_indices[0] += row_offset
        indices.append(batch_indices)
        values.append(coalesced.values())
        row_offset += coalesced.shape[0]

    return torch.sparse_coo_tensor(
        torch.cat(indices, dim=1),
        torch.cat(values),
        size=(row_offset, sparse_dim),
        dtype=values[0].dtype,
    ).coalesce()


class _UEmbedInference:
    """Inference core adapted from Alibaba-NLP/UEmbed.

    This adapter intentionally exposes only dense ``last.normal`` and sparse
    ``splade.last`` pooling.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_name_or_path: str,
        *,
        revision: str | None,
        pooling: Pooling,
        device: str | torch.device | None,
        normalize: bool,
        max_length: int,
        min_pixels: int,
        max_pixels: int,
        total_pixels: int,
        fps: float,
        max_frames: int,
        default_instruction: str,
        processor_kwargs: dict[str, Any] | None,
        **model_kwargs: Any,
    ) -> None:
        if pooling not in SUPPORTED_POOLING:
            raise ValueError(
                f"Unsupported UEmbed pooling {pooling!r}. "
                f"Choose one of {SUPPORTED_POOLING}."
            )

        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()

        processor_load_kwargs = dict(processor_kwargs or {})
        for key, value in _hub_download_kwargs(model_kwargs).items():
            processor_load_kwargs.setdefault(key, value)
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            revision=revision,
            padding_side="right",
            **processor_load_kwargs,
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"

        self.num_eos_tokens = 0
        self.sparse_lm_heads: torch.nn.ParameterList | None = None
        self.sparse_bias: torch.nn.ParameterList | None = None
        self.sparse_dim = SPARSE_DIM
        self._load_sparse_info(
            model_name_or_path,
            revision=revision,
            model_kwargs=model_kwargs,
        )
        if pooling == "splade.last":
            self._load_sparse_weights(
                model_name_or_path,
                revision=revision,
                model_kwargs=model_kwargs,
            )

        text_config = getattr(self.model.config, "text_config", self.model.config)
        self.dense_dim = int(text_config.hidden_size)

    @staticmethod
    def _resolve_artifact(
        model_name_or_path: str,
        filename: str,
        *,
        revision: str | None,
        model_kwargs: dict[str, Any],
    ) -> Path:
        local_path = Path(model_name_or_path)
        if local_path.is_dir():
            artifact = local_path / filename
            if not artifact.is_file():
                raise FileNotFoundError(
                    f"Required UEmbed artifact not found: {artifact}"
                )
            return artifact
        return Path(
            hf_hub_download(
                repo_id=model_name_or_path,
                filename=filename,
                revision=revision,
                **_hub_download_kwargs(model_kwargs),
            )
        )

    def _load_sparse_info(
        self,
        model_name_or_path: str,
        *,
        revision: str | None,
        model_kwargs: dict[str, Any],
    ) -> None:
        info_path = self._resolve_artifact(
            model_name_or_path,
            "sparse_info.json",
            revision=revision,
            model_kwargs=model_kwargs,
        )
        with info_path.open(encoding="utf-8") as file:
            sparse_info = json.load(file)
        self.num_eos_tokens = int(sparse_info.get("num_eos_tokens", 0))
        if self.num_eos_tokens <= 0:
            raise ValueError(
                f"Invalid num_eos_tokens in UEmbed sparse config: {self.num_eos_tokens}"
            )

    def _load_sparse_weights(
        self,
        model_name_or_path: str,
        *,
        revision: str | None,
        model_kwargs: dict[str, Any],
    ) -> None:
        weights_path = self._resolve_artifact(
            model_name_or_path,
            "sparse_weights.pt",
            revision=revision,
            model_kwargs=model_kwargs,
        )
        sparse_weights = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(sparse_weights, dict):
            raise ValueError("sparse_weights.pt must contain a dictionary")
        heads = sparse_weights.get("sparse_lm_heads")
        biases = sparse_weights.get("sparse_bias")
        if not isinstance(heads, (list, tuple)) or not isinstance(
            biases, (list, tuple)
        ):
            raise ValueError(
                "sparse_weights.pt must contain sparse_lm_heads and sparse_bias lists"
            )
        if len(heads) != self.num_eos_tokens or len(biases) != self.num_eos_tokens:
            raise ValueError(
                "UEmbed sparse head count does not match num_eos_tokens: "
                f"heads={len(heads)}, biases={len(biases)}, "
                f"tokens={self.num_eos_tokens}"
            )

        model_dtype = next(self.model.parameters()).dtype
        self.sparse_lm_heads = torch.nn.ParameterList(
            [
                torch.nn.Parameter(head.to(model_dtype), requires_grad=False)
                for head in heads
            ]
        ).to(self.device)
        self.sparse_bias = torch.nn.ParameterList(
            [
                torch.nn.Parameter(bias.to(model_dtype), requires_grad=False)
                for bias in biases
            ]
        ).to(self.device)
        self.sparse_dim = sum(int(head.shape[0]) for head in heads)

    @staticmethod
    def _format_instruction(instruction: str | None) -> str | None:
        if not instruction:
            return instruction
        instruction = instruction.strip()
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction += "."
        return instruction

    def _format_model_input(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        from PIL import Image

        instruction = self._format_instruction(item.get("instruction"))
        content: list[dict[str, Any]] = []
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": instruction or self.default_instruction,
                    }
                ],
            },
            {"role": "user", "content": content},
        ]

        text = item.get("text")
        texts = text if isinstance(text, list) else ([text] if text is not None else [])
        image = item.get("image")
        images = (
            image if isinstance(image, list) else ([image] if image is not None else [])
        )
        video = item.get("video")
        videos = [video] if video is not None else []

        if not texts and not images and not videos:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        for video_item in videos:
            if isinstance(video_item, list):
                frames = video_item[: self.max_frames]
                video_content = [
                    f"file://{frame}" if isinstance(frame, str) else frame
                    for frame in frames
                ]
                content.append(
                    {
                        "type": "video",
                        "video": video_content,
                        "total_pixels": self.total_pixels,
                    }
                )
            elif isinstance(video_item, str):
                video_content = (
                    video_item
                    if video_item.startswith(("http://", "https://"))
                    else f"file://{video_item}"
                )
                content.append(
                    {
                        "type": "video",
                        "video": video_content,
                        "fps": self.fps,
                        "max_frames": self.max_frames,
                    }
                )
            else:
                raise TypeError(f"Unrecognized video type: {type(video_item)}")

        for image_item in images:
            if isinstance(image_item, Image.Image):
                image_content = image_item
            elif isinstance(image_item, str):
                image_content = (
                    image_item
                    if image_item.startswith(("http://", "https://"))
                    else f"file://{image_item}"
                )
            else:
                raise TypeError(f"Unrecognized image type: {type(image_item)}")
            content.append(
                {
                    "type": "image",
                    "image": image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )

        content.extend(
            {"type": "text", "text": text_item} for text_item in texts if text_item
        )
        return conversation

    def _preprocess_inputs(
        self, conversations: list[list[dict[str, Any]]]
    ) -> dict[str, torch.Tensor]:
        from qwen_vl_utils.vision_process import process_vision_info

        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        images, video_inputs, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos = None
            video_metadata = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _pool_dense(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_indices = (attention_mask.cumsum(dim=1) * attention_mask).argmax(dim=1)
        target_indices = last_indices - self.num_eos_tokens
        batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        embeddings = hidden_state[batch_indices, target_indices]
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _pool_sparse(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.sparse_lm_heads is None or self.sparse_bias is None:
            raise RuntimeError("UEmbed sparse heads were not loaded")

        last_indices = (attention_mask.cumsum(dim=1) * attention_mask).argmax(dim=1)
        batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        logits = []
        for index in range(self.num_eos_tokens):
            offset = self.num_eos_tokens - 1 - index
            token_hidden_state = hidden_state[
                batch_indices,
                last_indices - offset,
            ]
            logits.append(
                F.linear(
                    token_hidden_state,
                    self.sparse_lm_heads[index],
                    self.sparse_bias[index],
                )
            )
        return torch.log1p(F.relu(torch.cat(logits, dim=-1)))

    @torch.no_grad()
    def process(self, items: list[dict[str, Any]]) -> torch.Tensor:
        conversations = [self._format_model_input(item) for item in items]
        inputs = self._preprocess_inputs(conversations)
        outputs = self.model(**inputs)
        hidden_state = outputs.last_hidden_state
        attention_mask = cast("torch.Tensor", inputs["attention_mask"])
        if self.pooling == "last.normal":
            return self._pool_dense(hidden_state, attention_mask)
        return self._pool_sparse(hidden_state, attention_mask)


class UEmbedEncoder(AbsEncoder):
    """MTEB adapter for UEmbed dense and sparse retrieval embeddings."""

    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        revision: str | None = None,
        *,
        device: str | None = None,
        pooling: Pooling = "last.normal",
        normalize: bool = True,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = DEFAULT_FPS,
        max_frames: int = DEFAULT_MAX_FRAMES,
        default_instruction: str = DEFAULT_INSTRUCTION,
        apply_instruction_to_passages: bool = False,
        processor_kwargs: dict[str, Any] | None = None,
        embed_dim: int | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.pooling = pooling
        self.apply_instruction_to_passages = apply_instruction_to_passages
        self.fps = fps
        self.max_frames = max_frames
        self._mteb_model_meta: ModelMeta | None = None
        model_kwargs.setdefault("dtype", torch.bfloat16)
        self.model = _UEmbedInference(
            model_name,
            revision=revision,
            pooling=pooling,
            device=device,
            normalize=normalize,
            max_length=max_length,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            fps=fps,
            max_frames=max_frames,
            default_instruction=default_instruction,
            processor_kwargs=processor_kwargs,
            **model_kwargs,
        )

        actual_dim = (
            self.model.dense_dim if pooling == "last.normal" else self.model.sparse_dim
        )
        if embed_dim is not None and embed_dim != actual_dim:
            raise ValueError(
                f"UEmbed pooling {pooling!r} produces dimension {actual_dim}, "
                f"not requested dimension {embed_dim}."
            )

    @property
    def mteb_model_meta(self) -> ModelMeta | None:
        return self._mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, meta: ModelMeta | None) -> None:
        if meta is None:
            self._mteb_model_meta = None
            return

        experiment_kwargs = dict(meta.experiment_kwargs or {})
        experiment_kwargs["pooling"] = self.pooling
        updates: dict[str, Any]
        if self.pooling == "last.normal":
            updates = {
                "model_type": ["dense"],
                "embed_dim": self.model.dense_dim,
                "similarity_fn_name": ScoringFunction.COSINE,
            }
        else:
            updates = {
                "model_type": ["sparse"],
                "embed_dim": self.model.sparse_dim,
                "similarity_fn_name": ScoringFunction.DOT_PRODUCT,
            }
        updates["experiment_kwargs"] = experiment_kwargs
        self._mteb_model_meta = meta.model_copy(update=updates, deep=True)

    def _instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str | None:
        if (
            prompt_type == PromptType.document
            and not self.apply_instruction_to_passages
        ):
            return None
        try:
            return self.get_task_instruction(task_metadata, prompt_type)
        except KeyError:
            # Tasks with no prompt in their metadata send AbsEncoder looking them up by
            # name, which fails for unregistered tasks such as mteb's mock tasks.
            return None

    @staticmethod
    def _batch_to_items(
        batch: BatchedInput,
        instruction: str | None,
    ) -> list[dict[str, Any]]:
        modality_keys = [key for key in ("text", "image", "video") if key in batch]
        if not modality_keys:
            raise ValueError(
                "UEmbed supports text, image, and video inputs; received no supported "
                f"modality in {list(batch)}"
            )

        batch_size = len(batch[modality_keys[0]])  # type: ignore[literal-required]
        items = []
        for row_index in range(batch_size):
            item: dict[str, Any] = {}
            for key in modality_keys:
                value = batch[key][row_index]  # type: ignore[index,literal-required]
                if key == "video" and isinstance(value, torch.Tensor):
                    value = _video_tensor_to_pil_frames(value)
                item[key] = value
            if instruction:
                item["instruction"] = instruction
            items.append(item)
        return items

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
        del hf_split, hf_subset, kwargs
        if self.pooling == "splade.last" and (
            task_metadata.simplified_task_type != "retrieval"
            or "Reranking" in task_metadata.type
        ):
            raise ValueError(
                "UEmbed splade.last currently supports retrieval tasks only; "
                f"task {task_metadata.name!r} has type {task_metadata.type!r}."
            )

        instruction = self._instruction(task_metadata, prompt_type)
        has_video = "video" in inputs.dataset.features  # type: ignore[attr-defined]
        original_collate_fn = inputs.collate_fn
        if has_video:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=16_000,
                fps=self.fps,
                max_frames=self.max_frames,
            )

        dense_batches: list[torch.Tensor] = []
        sparse_batches: list[torch.Tensor] = []
        try:
            for batch in tqdm(
                inputs,
                desc=f"Encoding UEmbed ({self.pooling})",
                disable=not show_progress_bar,
            ):
                items = self._batch_to_items(batch, instruction)
                embeddings = self.model.process(items).detach().float().cpu()
                if self.pooling == "last.normal":
                    dense_batches.append(embeddings)
                else:
                    sparse_batches.append(embeddings.to_sparse_coo().coalesce())
        finally:
            inputs.collate_fn = original_collate_fn

        if self.pooling == "last.normal":
            if not dense_batches:
                return torch.empty((0, self.model.dense_dim), dtype=torch.float32)
            return torch.cat(dense_batches)
        return _concatenate_sparse_batches(sparse_batches, self.model.sparse_dim)

    def similarity(self, embeddings1: Array, embeddings2: Array) -> torch.Tensor:
        if self.pooling == "last.normal":
            first = torch.as_tensor(embeddings1)
            second = torch.as_tensor(embeddings2)
            # Callers such as the summarization evaluator pass a single embedding per
            # side, which has no dimension to transpose.
            if first.ndim == 1:
                first = first.unsqueeze(0)
            if second.ndim == 1:
                second = second.unsqueeze(0)
            return first @ second.transpose(-2, -1)

        first = cast("torch.Tensor", embeddings1).coalesce()
        second = cast("torch.Tensor", embeddings2).coalesce()
        return torch.sparse.mm(first, second.transpose(0, 1)).to_dense()

    def similarity_pairwise(
        self, embeddings1: Array, embeddings2: Array
    ) -> torch.Tensor:
        if self.pooling == "last.normal":
            first = torch.as_tensor(embeddings1)
            second = torch.as_tensor(embeddings2)
            return (first * second).sum(dim=-1)

        first = cast("torch.Tensor", embeddings1).coalesce()
        second = cast("torch.Tensor", embeddings2).coalesce()
        return torch.sparse.sum(first * second, dim=1).to_dense()


UEMBED_CITATION = """@misc{uembed2026,
  title={UEmbed: Unified Sparse and Dense Multimodal Embeddings},
  author={Tingyu Song and Mingxin Li and Yanzhao Zhang and Dingkun Long and Pengjun Xie and Zhijie Nie and Yilun Zhao and Shu Wu},
  year={2026},
  eprint={2608.02583},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.02583},
}"""

UEMBED_TRAINING_DATASETS = {
    "MultiLongDocRetrieval",  # MLDR
    # The corpora below have no direct MTEB equivalent. MMEB-train overlaps several
    # MTEB image test sets, and the MMEB-V2 video and visual-document splits were
    # used, so image, video, and visual-document scores are in-domain.
    # "Echo-Embedding",
    # "MMEB-V1",
    # "MMEB-V2",
}

_COMMON_METADATA = dict(
    loader=UEmbedEncoder,
    model_type=["dense", "sparse"],
    languages=["eng-Latn"],
    open_weights=True,
    release_date="2026-08-04",
    modalities=["image", "text", "video"],
    license="cc-by-4.0",
    max_tokens=8192,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code="https://github.com/Alibaba-NLP/UEmbed",
    public_training_data=True,
    training_datasets=UEMBED_TRAINING_DATASETS,
    citation=UEMBED_CITATION,
    experiment_kwargs={"pooling": "last.normal"},
    extra_requirements_groups=["uembed"],
)

uembed_2b = ModelMeta(
    name="Alibaba-NLP/UEmbed-2B",
    revision="b544aa19b50b5961f0a14f22c9f8abf1eafa80bc",
    n_parameters=2_590_290_448,
    n_embedding_parameters=508_559_360,
    memory_usage_mb=4941,
    embed_dim=[2048, 184_016],
    reference="https://huggingface.co/Alibaba-NLP/UEmbed-2B",
    **_COMMON_METADATA,
)

uembed_4b = ModelMeta(
    name="Alibaba-NLP/UEmbed-4B",
    revision="5ceafb64ad1049f2c6d6b179dd8a878e75444a4f",
    n_parameters=5_010_530_512,
    n_embedding_parameters=635_699_200,
    memory_usage_mb=9557,
    embed_dim=[2560, 184_016],
    reference="https://huggingface.co/Alibaba-NLP/UEmbed-4B",
    **_COMMON_METADATA,
)

uembed_9b = ModelMeta(
    name="Alibaba-NLP/UEmbed-9B",
    revision="ea380efb495d62d324c7bbaa6c6b23a0f748692d",
    n_parameters=9_146_608_576,
    n_embedding_parameters=1_017_118_720,
    memory_usage_mb=17446,
    embed_dim=[4096, 184_016],
    reference="https://huggingface.co/Alibaba-NLP/UEmbed-9B",
    **_COMMON_METADATA,
)
