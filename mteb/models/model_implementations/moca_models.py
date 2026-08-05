from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING, Any

import torch
import transformers
from tqdm.auto import tqdm

from mteb._requires_package import suggest_package
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType

logger = logging.getLogger(__name__)

MOCA_CITATION = """@article{chen2025moca,
  title={MoCa: Modality-aware Continual Pre-training Makes Better Bidirectional Multimodal Embeddings},
  author={Chen, Haonan and Liu, Hong and Luo, Yuping and Wang, Liang and Yang, Nan and Wei, Furu and Dou, Zhicheng},
  journal={arXiv preprint arXiv:2506.23115},
  year={2025}
}"""

# Qwen2.5-VL image placeholder, as used by MoCa's collator
# (https://github.com/haon-chen/MoCa/blob/main/src/model_utils.py).
_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"  # noqa: S105

# Default prompts taken from the MoCa model card and evaluation code.
_IMAGE_PROMPT = "Represent the given document image."
_IMAGE_TEXT_PROMPT = "Represent the given image with the following question: {text}"

# Image resolution bounds used in MoCa's ViDoRe evaluation
# (https://github.com/haon-chen/MoCa/blob/main/evaluation/vidore_benchmark/moca_qwen25_retriever.py).
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28
MIN_IMAGE_SIDE = 28


def _bidirectional_causal_mask(
    self,
    attention_mask: torch.Tensor | None,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor | None = None,
    past_key_values: Any = None,
    output_attentions: bool = False,
) -> torch.Tensor | None:
    """Replacement for `Qwen2_5_VLTextModel._update_causal_mask` that only masks padding.

    MoCa turns the causal backbone into a bidirectional encoder. With
    `flash_attention_2` this is achieved by setting `is_causal=False` on every attention
    module, because the flash kernel reads that flag directly. The sdpa/eager paths
    instead build an explicit lower-triangular mask, so they need this patch to stay
    bidirectional.
    """
    if attention_mask is None:
        return None
    dtype = input_tensor.dtype
    min_dtype = torch.finfo(dtype).min
    padding_mask = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - padding_mask) * min_dtype


class MoCaWrapper(AbsEncoder):
    """Bidirectional multimodal embeddings from a Qwen2.5-VL backbone.

    Adapted from https://github.com/haon-chen/MoCa/blob/main/src/vlm_backbone/qwen2_5_vl_embed/qwen2_5_vl_embed.py.
    The reference implementation vendors a copy of `modeling_qwen2_5_vl.py`; the only
    functional differences from upstream `transformers` are (1) `is_causal = False` on
    the text attention modules and (2) mean pooling over the last hidden state instead
    of an LM head, both of which are reproduced here on the stock model class.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        processor_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        max_length: int = 2048,
        image_prompt: str = _IMAGE_PROMPT,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.image_prompt = image_prompt

        has_flash_attn = suggest_package(
            self,
            "flash_attn",
            model_name,
            "pip install flash-attn --no-build-isolation",
        )
        if attn_implementation is None:
            attn_implementation = (
                "flash_attention_2"
                if has_flash_attn and self.device.startswith("cuda")
                else "sdpa"
            )

        self.model, loading_info = Qwen2_5_VLModel.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            output_loading_info=True,
        )
        if loading_info["missing_keys"]:
            logger.warning(
                "%s was loaded with %d randomly initialized parameters (%s...). "
                "Embeddings will be meaningless - check that the checkpoint layout "
                "matches your installed transformers version.",
                model_name,
                len(loading_info["missing_keys"]),
                loading_info["missing_keys"][:3],
            )

        # MoCa removes causal masking from the language tower.
        for layer in self.model.language_model.layers:
            layer.self_attn.is_causal = False
        if attn_implementation != "flash_attention_2":
            logger.warning(
                "Running %s without flash attention. Results may differ slightly from "
                "the published numbers, which were produced with flash_attention_2.",
                model_name,
            )
            if not hasattr(self.model.language_model, "_update_causal_mask"):
                raise RuntimeError(
                    "Cannot disable causal masking on transformers "
                    f"{transformers.__version__} without flash attention: "
                    "Qwen2_5_VLTextModel._update_causal_mask no longer exists. "
                    "Install flash-attn, or pin transformers==4.52.4."
                )
            self.model.language_model._update_causal_mask = types.MethodType(
                _bidirectional_causal_mask, self.model.language_model
            )

        self.model.config.use_cache = False
        self.model.to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=True,
        )
        self.processor.tokenizer.padding_side = "right"

    @staticmethod
    def _prepare_image(image: Any) -> Any:
        image = image.convert("RGB")
        width, height = image.size
        if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
            image = image.resize(
                (max(width, MIN_IMAGE_SIDE), max(height, MIN_IMAGE_SIDE))
            )
        return image

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.inference_mode()
    def _embed(self, texts: list[str], images: list[Any] | None = None) -> torch.Tensor:
        """Encode one batch of (optionally interleaved) inputs into unit-norm vectors."""
        processed = self.processor(
            text=texts,
            images=images or None,
            padding=True,
            # truncating a sequence that contains image placeholders would
            # desynchronise the image tokens from the vision features
            truncation=images is None,
            max_length=None if images else self.max_length,
            return_tensors="pt",
        )
        processed = {k: v.to(self.device) for k, v in processed.items()}
        outputs = self.model(**processed, use_cache=False, return_dict=True)
        reps = self._mean_pool(outputs.last_hidden_state, processed["attention_mask"])
        return torch.nn.functional.normalize(reps, p=2, dim=-1)

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
        features = inputs.dataset.features
        has_text = "text" in features
        has_image = "image" in features
        if not (has_text or has_image):
            raise ValueError(f"{task_metadata.name} has no text or image column.")

        show_progress_bar = kwargs.get("show_progress_bar", True)
        all_embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Encoding"):
            batch_size = len(next(iter(batch.values())))
            texts: list[str] = []
            images: list[Any] = []

            for i in range(batch_size):
                text = batch["text"][i] if has_text else None
                if has_image:
                    images.append(self._prepare_image(batch["image"][i]))
                    if text:
                        texts.append(
                            _IMAGE_TOKEN + _IMAGE_TEXT_PROMPT.format(text=text) + "\n"
                        )
                    else:
                        texts.append(_IMAGE_TOKEN + self.image_prompt + "\n")
                else:
                    texts.append(text)

            reps = self._embed(texts, images if has_image else None)
            all_embeddings.append(reps.cpu().to(torch.float32))

        return torch.cat(all_embeddings, dim=0)


# MMEB-train appears in both the CPT and CL mixtures and overlaps the mteb tasks below,
# so scores on those should be read as in-domain. The other training corpora (DCLM,
# PixelProse, MAmmoTH-VL-Instruct, DocMatix, VisRAG, the ColPali training set, mmE5, E5)
# have no mteb equivalent.
moca_training_datasets = {
    "HatefulMemesI2TRetrieval",
    "HatefulMemesT2IRetrieval",
    "SUN397",
    "SUN397ZeroShot",
    "VOC2007",
    "OKVQAIT2TRetrieval",
    "CIRRIT2IRetrieval",
    "NIGHTSI2IRetrieval",
    "WebQAT2ITRetrieval",
    "WebQAT2TRetrieval",
    "VisualNewsI2TRetrieval",
    "VisualNewsT2IRetrieval",
    "MSCOCOI2TRetrieval",
    "MSCOCOT2IRetrieval",
}

moca_qwen25vl_3b = ModelMeta(
    loader=MoCaWrapper,
    loader_kwargs=dict(processor_name="Qwen/Qwen2.5-VL-3B-Instruct"),
    name="moca-embed/MoCa-Qwen25VL-3B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="89b0e5a9245a95d6df5f54e7ebe1588bc7c7d926",
    release_date="2025-06-29",
    modalities=["image", "text"],
    n_parameters=3_756_720_128,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=7165,
    embed_dim=2048,
    license="mit",
    max_tokens=32768,
    reference="https://huggingface.co/moca-embed/MoCa-Qwen25VL-3B",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code="https://github.com/haon-chen/MoCa",
    public_training_data="https://huggingface.co/moca-embed/datasets",
    training_datasets=moca_training_datasets,
    adapted_from="Qwen/Qwen2.5-VL-3B-Instruct",
    citation=MOCA_CITATION,
    extra_requirements_groups=["moca"],
)
