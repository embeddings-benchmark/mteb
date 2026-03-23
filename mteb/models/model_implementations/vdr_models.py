from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return "{instruction}"


class VDRModel(InstructSentenceTransformerModel):
    """SentenceTransformer wrapper with image/text support for VDR."""

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
        instruction = self.get_task_instruction(task_metadata, prompt_type)
        if (
            not self.apply_instruction_to_passages
            and prompt_type == PromptType.document
        ):
            instruction = None

        text_embeddings = None
        image_embeddings = None

        if "text" in inputs.dataset.features:
            texts = [text for batch in inputs for text in batch["text"]]
            text_embeddings = self.model.encode(texts, prompt=instruction, **kwargs)

        if "image" in inputs.dataset.features:
            images = [image for batch in inputs for image in batch["image"]]
            image_embeddings = self.model.encode(images, **kwargs)

        if isinstance(text_embeddings, torch.Tensor):
            text_embeddings = text_embeddings.cpu().detach().float().numpy()
        if isinstance(image_embeddings, torch.Tensor):
            image_embeddings = image_embeddings.cpu().detach().float().numpy()

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
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
        instruction_template=instruction_template,
        max_seq_length=32768,
        apply_instruction_to_passages=True,
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
