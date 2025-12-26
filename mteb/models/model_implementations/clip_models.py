from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
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
    n_parameters=428_000_000,
    memory_usage_mb=1631,
    max_tokens=77,
    embed_dim=768,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
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
    n_parameters=151_000_000,
    memory_usage_mb=576,
    max_tokens=77,
    embed_dim=512,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
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
    n_parameters=151_000_000,
    memory_usage_mb=576,
    max_tokens=77,
    embed_dim=512,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/openai/clip-vit-base-patch16",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    citation=CLIP_CITATION,
)
