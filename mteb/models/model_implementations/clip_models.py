from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_utils import get_present_indices
from mteb.models.model_meta import ModelMeta, ScoringFunction

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class CLIPModel(AbsEncoder):
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
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(
            self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_outputs = self.model.get_text_features(**inputs)
        # Handle both tensor and BaseModelOutputWithPooling returns
        if hasattr(text_outputs, "pooler_output"):
            text_outputs = text_outputs.pooler_output
        return text_outputs.cpu()

    @torch.no_grad()
    def _encode_images(self, images: list[Any]) -> torch.Tensor:
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_outputs = self.model.get_image_features(**inputs)
        # Handle both tensor and BaseModelOutputWithPooling returns
        if hasattr(image_outputs, "pooler_output"):
            image_outputs = image_outputs.pooler_output
        return image_outputs.cpu()

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_text_embeddings = [
            self._encode_texts(batch["text"])
            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            )
        ]
        return torch.cat(all_text_embeddings, dim=0)

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_image_embeddings = [
            self._encode_images(batch["image"])
            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            )
        ]
        return torch.cat(all_image_embeddings, dim=0)

    def get_fused_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sum the text and image embedding of every row.

        The dataset may interleave modalities, so a row can carry only a text or
        only an image; each row is the sum over the modalities it actually carries.
        A row carrying neither is embedded as zeros.
        """
        all_embeddings = []
        for batch in tqdm(
            inputs, disable=not show_progress_bar, desc="Interleaved Encoding"
        ):
            text_rows = get_present_indices(batch, "text")
            image_rows = get_present_indices(batch, "image")
            batch_size = len(batch["text"])

            encoded = []
            if text_rows:
                encoded.append(
                    (
                        text_rows,
                        self._encode_texts([batch["text"][i] for i in text_rows]),
                    )
                )
            if image_rows:
                encoded.append(
                    (
                        image_rows,
                        self._encode_images([batch["image"][i] for i in image_rows]),
                    )
                )
            if not encoded:
                raise ValueError(
                    "Batch carries neither text nor images; nothing to encode."
                )
            covered = len(set(text_rows) | set(image_rows))
            if covered < batch_size:
                logger.warning(
                    "%d row(s) carry no modality at all and are embedded as zeros.",
                    batch_size - covered,
                )

            dim = encoded[0][1].shape[-1]
            fused = torch.zeros(batch_size, dim, dtype=encoded[0][1].dtype)
            for rows, vectors in encoded:
                fused[rows] += vectors
            all_embeddings.append(fused)
        return torch.cat(all_embeddings, dim=0)

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
        has_text = "text" in inputs.dataset.features
        has_image = "image" in inputs.dataset.features

        if has_text and has_image:
            return self.get_fused_embeddings(inputs, **kwargs)
        if has_text:
            return self.get_text_embeddings(inputs, **kwargs)
        if has_image:
            return self.get_image_embeddings(inputs, **kwargs)
        raise ValueError


CLIP_CITATION = """
@article{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}"""


clip_vit_large_patch14 = ModelMeta(
    loader=CLIPModel,
    name="openai/clip-vit-large-patch14",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
    release_date="2021-02-26",
    modalities=["image", "text"],
    n_parameters=427616513,
    n_embedding_parameters=37945344,
    memory_usage_mb=1631,
    max_tokens=77,
    embed_dim=768,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/openai/clip-vit-large-patch14",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    citation=CLIP_CITATION,
)

clip_vit_base_patch32 = ModelMeta(
    loader=CLIPModel,
    name="openai/clip-vit-base-patch32",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    release_date="2021-02-26",
    modalities=["image", "text"],
    n_parameters=151277313,
    n_embedding_parameters=25296896,
    memory_usage_mb=576,
    max_tokens=77,
    embed_dim=512,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/openai/clip-vit-base-patch32",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    citation=CLIP_CITATION,
)

clip_vit_base_patch16 = ModelMeta(
    loader=CLIPModel,
    name="openai/clip-vit-base-patch16",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
    release_date="2021-02-26",
    modalities=["image", "text"],
    n_parameters=149620737,
    n_embedding_parameters=25296896,
    memory_usage_mb=576,
    max_tokens=77,
    embed_dim=512,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/openai/clip-vit-base-patch16",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    citation=CLIP_CITATION,
)
