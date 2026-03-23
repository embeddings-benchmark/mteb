from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from PIL.Image import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput


class VDRModel(AbsEncoder):
    """Transformers-based wrapper for vdr-2b-multi-v1 text/image encoding."""

    document_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "What is shown in this image?<|im_end|>\n<|endoftext|>"
    )
    query_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Query: %s<|im_end|>\n<|endoftext|>"
    )

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        trust_remote_code: bool = True,
        max_pixels: int = 768 * 28 * 28,
        min_pixels: int = 1 * 28 * 28,
        apply_instruction_to_passages: bool = True,
        **kwargs: Any,
    ):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.apply_instruction_to_passages = apply_instruction_to_passages
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).to(self.device)
        self.model.eval()
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        except ValueError as e:
            if (
                "shortest_edge" in str(e)
                and "longest_edge" in str(e)
            ):
                longest_edge = max(56, int(math.sqrt(self.max_pixels)))
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    size={
                        "shortest_edge": 56,
                        "longest_edge": longest_edge,
                    },
                )
            else:
                raise
        self.model.padding_side = "left"
        if getattr(self.processor, "tokenizer", None):
            self.processor.tokenizer.padding_side = "left"

    @staticmethod
    def _round_by_factor(number: float, factor: int) -> int:
        return round(number / factor) * factor

    @staticmethod
    def _ceil_by_factor(number: float, factor: int) -> int:
        return math.ceil(number / factor) * factor

    @staticmethod
    def _floor_by_factor(number: float, factor: int) -> int:
        return math.floor(number / factor) * factor

    def _smart_resize(self, height: int, width: int) -> tuple[int, int]:
        h_bar = max(28, self._round_by_factor(height, 28))
        w_bar = max(28, self._round_by_factor(width, 28))
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = self._floor_by_factor(height / beta, 28)
            w_bar = self._floor_by_factor(width / beta, 28)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = self._ceil_by_factor(height * beta, 28)
            w_bar = self._ceil_by_factor(width * beta, 28)
        return w_bar, h_bar

    def _resize(self, image: Image) -> Image:
        new_size = self._smart_resize(image.height, image.width)
        return image.resize(new_size)

    def _move_to_device(self, processed: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key, value in processed.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(self.device)
            else:
                out[key] = value
        return out

    @torch.no_grad()
    def _encode(
        self,
        *,
        texts: list[str],
        images: list[Any],
        batch_size: int = 1,
    ) -> Array:
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            batch_images = images[start:end]
            processed = self.processor(
                text=batch_texts,
                images=batch_images,
                videos=None,
                padding="longest",
                return_tensors="pt",
            )
            processed = self._move_to_device(processed)
            cache_position = torch.arange(0, len(batch_texts)).to(self.device)
            processed = self.model.prepare_inputs_for_generation(
                **processed,
                cache_position=cache_position,
                use_cache=False,
            )
            output = self.model(
                **processed,
                return_dict=True,
                output_hidden_states=True,
            )
            embeddings = output.hidden_states[-1][:, -1]
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().detach().float().numpy()
            all_embeddings.append(embeddings)
        if not all_embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(all_embeddings, axis=0)

    def get_text_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        batch_size: int = 1,
    ) -> Array:
        texts = [text for batch in inputs for text in batch["text"]]
        try:
            from PIL import Image

            dummy_images = [Image.new("RGB", (56, 56)) for _ in texts]
        except ImportError:
            # Unit tests can run without image dependencies by using placeholders.
            dummy_images = [None for _ in texts]
        query_texts = [self.query_prompt % text for text in texts]
        return self._encode(texts=query_texts, images=dummy_images, batch_size=batch_size)

    def get_image_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        batch_size: int = 1,
    ) -> Array:
        images = [image for batch in inputs for image in batch["image"]]
        resized_images = [self._resize(image) for image in images]
        prompts = [self.document_prompt] * len(resized_images)
        return self._encode(texts=prompts, images=resized_images, batch_size=batch_size)

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
        batch_size = kwargs.get("batch_size", 1)
        text_embeddings = None
        image_embeddings = None

        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, batch_size=batch_size)
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, batch_size=batch_size)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError("The number of texts and images must have the same length")
            return text_embeddings + image_embeddings
        if text_embeddings is not None:
            return text_embeddings
        if image_embeddings is not None:
            return image_embeddings
        raise ValueError("No text or image features found in inputs")


vdr_languages = [
    "eng-Latn",
    "ita-Latn",
    "fra-Latn",
    "deu-Latn",
    "spa-Latn",
]

vdr_2b_multi_v1 = ModelMeta(
    loader=VDRModel,
    loader_kwargs=dict(
        apply_instruction_to_passages=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ),
    name="llamaindex/vdr-2b-multi-v1",
    model_type=["dense"],
    languages=vdr_languages,
    open_weights=True,
    revision="2c4e54c8db4071cc61fc3c62f4490124e40c37db",
    release_date="2024-01-08",
    modalities=["text", "image"],
    n_parameters=2208985600,
    n_embedding_parameters=233_373_696,
    memory_usage_mb=4213,
    max_tokens=32768,
    embed_dim=1536,
    license="apache-2.0",
    reference="https://huggingface.co/llamaindex/vdr-2b-multi-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Sentence Transformers", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/llamaindex/vdr-multilingual-train",
    training_datasets=set(
        # llamaindex/vdr-multilingual-train
        "VDRMultilingualRetrieval",
    ),
)
