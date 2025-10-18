from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

SIGLIP_CITATION = """@misc{zhai2023sigmoid,
      title={Sigmoid Loss for Language Image Pre-Training},
      author={Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
      year={2023},
      eprint={2303.15343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}"""


class SiglipModelWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from transformers import AutoModel, AutoProcessor

        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            raise ImportError(
                "The `sentencepiece` package is required to run `pip install sentencepiece`"
            )

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
                    padding="max_length",
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
        **kwargs: Any,
    ):
        all_image_embeddings = []

        with torch.no_grad():
            for batch in tqdm(images):
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


siglip_training_datasets = set(
    # WebLI https://arxiv.org/abs/2209.06794
)

siglip_so400m_patch14_224 = ModelMeta(
    loader=SiglipModelWrapper,  # type: ignore
    name="google/siglip-so400m-patch14-224",
    languages=["eng-Latn"],
    revision="d04cf29fca7b6374f74d8bea1969314492266b5e",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=877_000_000,
    memory_usage_mb=3347,
    max_tokens=16,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-so400m-patch14-224",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
)

siglip_so400m_patch14_384 = ModelMeta(
    loader=SiglipModelWrapper,  # type: ignore
    name="google/siglip-so400m-patch14-384",
    languages=["eng-Latn"],
    revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=878_000_000,
    memory_usage_mb=3349,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-so400m-patch14-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
)

siglip_so400m_patch16_256_i18n = ModelMeta(
    loader=SiglipModelWrapper,  # type: ignore
    name="google/siglip-so400m-patch16-256-i18n",
    languages=["eng-Latn"],
    revision="365d321c0cfdea96bc28e3a29787a11a062681a1",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=1_130_000_000,
    memory_usage_mb=4306,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-so400m-patch16-256-i18n",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
)

siglip_base_patch16_256_multilingual = ModelMeta(
    loader=SiglipModelWrapper,  # type: ignore
    name="google/siglip-base-patch16-256-multilingual",
    languages=["eng-Latn"],
    revision="8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=371_000_000,
    memory_usage_mb=1414,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-base-patch16-256-multilingual",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
)

siglip_base_patch16_256 = ModelMeta(
    loader=SiglipModelWrapper,  # type: ignore
    name="google/siglip-base-patch16-256",
    languages=["eng-Latn"],
    revision="b078df89e446d623010d890864d4207fe6399f61",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=203_000_000,
    memory_usage_mb=775,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-base-patch16-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
)

siglip_base_patch16_512 = ModelMeta(
    loader=SiglipModelWrapper,  # type: ignore
    name="google/siglip-base-patch16-512",
    languages=["eng-Latn"],
    revision="753a949581523b60257d93e18391e8c27f72eb22",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=204_000_000,
    memory_usage_mb=777,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-base-patch16-512",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
)

siglip_base_patch16_384 = ModelMeta(
    loader=SiglipModelWrapper,  # type: ignore
    name="google/siglip-base-patch16-384",
    languages=["eng-Latn"],
    revision="41aec1c83b32e0a6fca20ad88ba058aa5b5ea394",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=203_000_000,
    memory_usage_mb=776,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-base-patch16-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
)

siglip_base_patch16_224 = ModelMeta(
    loader=SiglipModelWrapper,  # type: ignore
    name="google/siglip-base-patch16-224",
    languages=["eng-Latn"],
    revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=203_000_000,
    memory_usage_mb=775,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-base-patch16-224",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
)

siglip_large_patch16_256 = ModelMeta(
    loader=SiglipModelWrapper,  # type: ignore
    name="google/siglip-large-patch16-256",
    languages=["eng-Latn"],
    revision="d0da9f876e7d66b4e250cd2450c3ba2ce735e447",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=652_000_000,
    memory_usage_mb=2488,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-large-patch16-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
)

siglip_large_patch16_384 = ModelMeta(
    loader=SiglipModelWrapper,  # type: ignore
    name="google/siglip-large-patch16-384",
    languages=["eng-Latn"],
    revision="ce005573a40965dfd21fd937fbdeeebf2439fc35",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=652_000_000,
    memory_usage_mb=2489,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/google/siglip-large-patch16-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
)
