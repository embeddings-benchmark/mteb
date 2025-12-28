from typing import Any

import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

BLIP_CITATION = """@misc{https://doi.org/10.48550/arxiv.2201.12086,
    doi = {10.48550/ARXIV.2201.12086},
    url = {https://arxiv.org/abs/2201.12086},
    author = {Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
    keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}"""


class BLIPModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from transformers import BlipForImageTextRetrieval, BlipProcessor

        self.model_name = model_name
        self.device = device
        self.model = BlipForImageTextRetrieval.from_pretrained(
            model_name, revision=revision
        ).to(self.device)
        self.processor = BlipProcessor.from_pretrained(model_name, revision=revision)

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
                # different to CLIPModelWrapper: text_encoder instead of get_text_features and apply projection and normalization
                text_outputs = self.model.text_encoder(**inputs)
                text_outputs = text_outputs[0]
                text_outputs = normalize(
                    self.model.text_proj(text_outputs[:, 0, :]), dim=-1
                )
                all_text_embeddings.append(text_outputs.cpu())

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
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
                image_outputs = self.model.vision_model(**inputs)
                image_outputs = image_outputs[0]
                image_outputs = normalize(
                    self.model.vision_proj(image_outputs[:, 0, :]), dim=-1
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


# in descending order of usage (downloads from huggingface)
blip_image_captioning_large = ModelMeta(
    loader=BLIPModel,
    name="Salesforce/blip-image-captioning-large",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="2227ac38c9f16105cb0412e7cab4759978a8fd90",
    release_date="2023-12-07",
    modalities=["image", "text"],
    n_parameters=470_000_000,
    memory_usage_mb=1792,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-image-captioning-large",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        # COCO
        # CC3M+CC12M+SBU
        # LAION115M
    ),
    citation=BLIP_CITATION,
)

blip_image_captioning_base = ModelMeta(
    loader=BLIPModel,
    name="Salesforce/blip-image-captioning-base",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="89b09ea1789f7addf2f6d6f0dfc4ce10ab58ef84",
    release_date="2023-08-01",
    modalities=["image", "text"],
    n_parameters=247_000_000,
    memory_usage_mb=942,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-image-captioning-base",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        # COCO
        # CC3M+CC12M+SBU
        # LAION115M
    ),
    citation=BLIP_CITATION,
)


blip_vqa_base = ModelMeta(
    loader=BLIPModel,
    name="Salesforce/blip-vqa-base",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="c7df8e7cd7aa2ee9af18f56e2b29e59a92651b64",
    release_date="2023-12-07",
    modalities=["image", "text"],
    n_parameters=247_000_000,
    memory_usage_mb=1467,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-vqa-base",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        # CC3M+CC12M+SBU
        # LAION115M
    ),
    citation=BLIP_CITATION,
)

blip_vqa_capfilt_large = ModelMeta(
    loader=BLIPModel,
    name="Salesforce/blip-vqa-capfilt-large",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="e53f95265aeab69013fabb5380500ab984adbbb4",
    release_date="2023-01-22",
    modalities=["image", "text"],
    n_parameters=247_000_000,
    memory_usage_mb=942,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-vqa-capfilt-large",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        # CC3M+CC12M+SBU
        # LAION115M
    ),
    citation=BLIP_CITATION,
)

blip_itm_base_coco = ModelMeta(
    loader=BLIPModel,
    name="Salesforce/blip-itm-base-coco",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="7eaa90c11850c0b17fc38c6a11e7d88bd6ac231f",
    release_date="2023-08-01",
    modalities=["image", "text"],
    n_parameters=247_000_000,
    memory_usage_mb=942,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-itm-base-coco",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        # CC3M+CC12M+SBU
        # LAION115M
    ),
    citation=BLIP_CITATION,
)

blip_itm_large_coco = ModelMeta(
    loader=BLIPModel,
    name="Salesforce/blip-itm-large-coco",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="fef05cafc05298067cbbca00b125749394a77a6f",
    release_date="2023-08-01",
    modalities=["image", "text"],
    n_parameters=470_000_000,
    memory_usage_mb=1793,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-itm-large-coco",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        # COCO
        # CC3M+CC12M+SBU
        # LAION115M
    ),
    citation=BLIP_CITATION,
)

blip_itm_base_flickr = ModelMeta(
    loader=BLIPModel,
    name="Salesforce/blip-itm-base-flickr",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="1de29e660d91ae1786c1876212ea805a22eab251",
    release_date="2023-08-01",
    modalities=["image", "text"],
    n_parameters=247_000_000,
    memory_usage_mb=942,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-itm-base-flickr",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        # CC3M+CC12M+SBU
        # LAION115M
        # Flickr30k
    ),
    citation=BLIP_CITATION,
)

blip_itm_large_flickr = ModelMeta(
    loader=BLIPModel,
    name="Salesforce/blip-itm-large-flickr",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="bda12e6506758f54261b5ab174b2c55a3ba143fb",
    release_date="2023-08-01",
    modalities=["image", "text"],
    n_parameters=470_000_000,
    memory_usage_mb=1793,
    max_tokens=512,
    embed_dim=768,
    license="bsd-3-clause",
    open_weights=True,
    public_training_code="https://github.com/salesforce/BLIP",
    public_training_data="https://github.com/salesforce/BLIP",
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip-itm-large-flickr",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        # CC3M+CC12M+SBU
        # LAION115M
    ),
    citation=BLIP_CITATION,
)
