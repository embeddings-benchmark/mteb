from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Unpack

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


class VDRModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str | None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.max_pixels = 768 * 28 * 28
        self.min_pixels = 1 * 28 * 28

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
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        self.model.padding_side = "left"
        self.processor.tokenizer.padding_side = "left"

        self.image_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|endoftext|>"
        self.text_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: %s<|im_end|>\n<|endoftext|>"

    @torch.no_grad()
    def get_text_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        all_text_embeddings = []
        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Text Encoding",
        ):
            inputs = self.processor(
                text=[self.text_prompt % text for text in batch["text"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            cache_position = torch.arange(0, len(inputs["input_ids"]))
            inputs = self.model.prepare_inputs_for_generation(
                **inputs,
                cache_position=cache_position,
                use_cache=False,
            )
            output = self.model(**inputs, return_dict=True, output_hidden_states=True)
            embeddings = output.hidden_states[-1][:, -1]
            all_text_embeddings.append(embeddings.cpu())
        return torch.cat(all_text_embeddings, dim=0)

    def round_by_factor(self, number: float, factor: int) -> int:
        return round(number / factor) * factor

    def ceil_by_factor(self, number: float, factor: int) -> int:
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: float, factor: int) -> int:
        return math.floor(number / factor) * factor

    def smart_resize(self, height: int, width: int) -> tuple[int, int]:
        h_bar = max(28, self.round_by_factor(height, 28))
        w_bar = max(28, self.round_by_factor(width, 28))
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = self.floor_by_factor(height / beta, 28)
            w_bar = self.floor_by_factor(width / beta, 28)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, 28)
            w_bar = self.ceil_by_factor(width * beta, 28)
        return w_bar, h_bar

    def resize(self, image: Image.Image):
        new_size = self.smart_resize(image.height, image.width)
        return image.resize(new_size)

    @torch.no_grad()
    def get_image_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        all_image_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
            desc="Image Encoding",
        ):
            inputs = self.processor(
                text=[self.image_prompt] * len(batch["image"]),
                images=[self.resize(x) for x in batch["image"]],
                videos=None,
                padding="longest",
                return_tensors="pt",
            ).to(self.device)

            cache_position = torch.arange(0, len(inputs["input_ids"]))
            inputs = self.model.prepare_inputs_for_generation(
                **inputs,
                cache_position=cache_position,
                use_cache=False,
            )
            output = self.model(**inputs, return_dict=True, output_hidden_states=True)
            embeddings = output.hidden_states[-1][:, -1]
            all_image_embeddings.append(embeddings.cpu())
        return torch.cat(all_image_embeddings, dim=0)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        text_embeddings = None
        image_embeddings = None
        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            fused_embeddings = text_embeddings + image_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError


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
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
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
    adapted_from="MrLight/dse-qwen2-2b-mrl-v1",
)
