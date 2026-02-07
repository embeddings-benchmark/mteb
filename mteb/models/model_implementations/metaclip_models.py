from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

from .facebookai import XLMR_LANGUAGES

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


METACLIP2_CITATION = """@article{xu2025metaclip2,
  title={MetaCLIP 2: A Worldwide Scaling Recipe},
  author={Xu, Hu and Xie, Saining and Ghosh, Gargi and Kira, Zsolt and Darrell, Trevor},
  journal={arXiv preprint arXiv:2507.22062},
  year={2025}
}"""


class MetaClip2Model(AbsEncoder):
    """Wrapper for MetaCLIP 2 models.

    MetaCLIP 2 is a multilingual vision-language model that uses the mT5 tokenizer
    for worldwide language support.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
        )

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                inputs = self.processor(
                    text=batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_outputs = self.model.get_text_features(**inputs)
                # MetaCLIP 2 returns BaseModelOutputWithPooling, extract pooler_output
                if hasattr(text_outputs, "pooler_output"):
                    text_outputs = text_outputs.pooler_output
                all_text_embeddings.append(text_outputs.cpu())

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

    @torch.no_grad()
    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_image_embeddings = []

        for batch in tqdm(images, disable=not show_progress_bar, desc="Image Encoding"):
            inputs = self.processor(
                images=batch["image"],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_outputs = self.model.get_image_features(**inputs)
            # MetaCLIP 2 returns BaseModelOutputWithPooling, extract pooler_output
            if hasattr(image_outputs, "pooler_output"):
                image_outputs = image_outputs.pooler_output
            all_image_embeddings.append(image_outputs.cpu())

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings

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


metaclip2_mt5_worldwide_b32 = ModelMeta(
    loader=MetaClip2Model,
    name="facebook/metaclip-2-mt5-worldwide-b32",
    model_type=["dense"],
    languages=XLMR_LANGUAGES,
    revision="fbbce525749bfc4a54b932bafe85313ee889d98f",
    release_date="2025-11-12",
    modalities=["image", "text"],
    n_parameters=253980417,
    n_embedding_parameters=128_057_344,
    memory_usage_mb=969,
    max_tokens=77,
    embed_dim=512,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/MetaCLIP",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/facebook/metaclip-2-mt5-worldwide-b32",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets={"CommonCrawl"},
    citation=METACLIP2_CITATION,
)
