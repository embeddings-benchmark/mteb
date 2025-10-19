from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import requires_image_dependencies
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

JINA_CLIP_CITATION = """@article{koukounas2024jinaclip,
  title={Jina CLIP: Your CLIP Model Is Also Your Text Retriever},
  author={Koukounas, Andreas and Mastrapas, Georgios and Günther, Michael and Wang, Bo and Martens, Scott and Mohr, Isabelle and Sturua, Saba and Akram, Mohammad Kalim and Martínez, Joan Fontanals and Ognawala, Saahil and Guzman, Susana and Werk, Maximilian and Wang, Nan and Xiao, Han},
  journal={arXiv preprint arXiv:2405.20204},
  year={2024}
}"""


class JinaCLIPModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from transformers import AutoModel

        requires_image_dependencies()

        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        ).to(self.device)

    def get_text_embeddings(
        self,
        texts: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = True,
        **kwargs: Any,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                texts, disable=not show_progress_bar, desc="Text Encoding"
            ):
                text_outputs = self.model.encode_text(
                    batch["text"],
                    convert_to_numpy=convert_to_numpy,
                    convert_to_tensor=convert_to_tensor,
                )
                all_text_embeddings.append(text_outputs.cpu())

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        *,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = True,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_image_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                images, disable=not show_progress_bar, desc="Image Encoding"
            ):
                image_outputs = self.model.encode_image(
                    batch["image"],
                    convert_to_numpy=convert_to_numpy,
                    convert_to_tensor=convert_to_tensor,
                )
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
            text_embeddings = self.get_text_embeddings(
                inputs, convert_to_numpy=False, convert_to_tensor=True, **kwargs
            )
        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(
                inputs, convert_to_numpy=False, convert_to_tensor=True, **kwargs
            )

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


jina_clip_v1 = ModelMeta(
    loader=JinaCLIPModel,  # type: ignore
    name="jinaai/jina-clip-v1",
    languages=["eng-Latn"],
    revision="06150c7c382d7a4faedc7d5a0d8cdb59308968f4",
    release_date="2024-05-30",
    modalities=["image", "text"],
    n_parameters=223_000_000,
    memory_usage_mb=849,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/jinaai/jina-clip-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets={
        # LAION400M
        # ShareGPT4V
        "MSMARCO",
        # NQ
        # HotpotQA
        # Natural Language Inference (NLI) dataset (Bowman et al., 2015)
    },
    citation=JINA_CLIP_CITATION,
)
