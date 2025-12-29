from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType


class ALIGNModel(AbsEncoder):
    def __init__(
        self,
        model: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from transformers import AutoModel, AutoProcessor

        self.model_name = model
        self.device = device
        self.model = AutoModel.from_pretrained(model, revision=revision).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model, revision=revision)

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs,
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
                all_text_embeddings.append(text_outputs.cpu())

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs,
    ):
        all_image_embeddings = []
        with torch.no_grad():
            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                inputs = self.processor(
                    images=batch["image"], return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_outputs = self.model.get_image_features(**inputs)
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


align_base = ModelMeta(
    loader=ALIGNModel,
    name="kakaobrain/align-base",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="e96a37facc7b1f59090ece82293226b817afd6ba",
    release_date="2023-02-24",
    modalities=["image", "text"],
    n_parameters=176_000_000,
    memory_usage_mb=671,
    max_tokens=64,
    embed_dim=768,
    license=None,
    open_weights=True,
    public_training_code="https://github.com/kakaobrain/coyo-align",
    public_training_data=True,
    framework=["PyTorch"],
    reference="https://huggingface.co/kakaobrain/align-base",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        #  COYO-700M
    ),
    citation="""@misc{kakaobrain2022coyo-align,
    title         = {COYO-ALIGN},
    author        = {Yoon, Boogeo and Lee, Youhan and Baek, Woonhyuk},
    year          = {2022},
    howpublished  = {https://github.com/kakaobrain/coyo-align},
}""",
)
