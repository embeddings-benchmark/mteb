from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutputWithPooling

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
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
                if isinstance(text_outputs, BaseModelOutputWithPooling):
                    embeddings = text_outputs.pooler_output
                else:
                    embeddings = text_outputs
                all_text_embeddings.append(embeddings.cpu())

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
                _images = [img.convert("RGB") for img in batch["image"]]
                inputs = self.processor(
                    images=_images, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_outputs = self.model.get_image_features(**inputs)
                if isinstance(image_outputs, BaseModelOutputWithPooling):
                    embeddings = image_outputs.pooler_output
                else:
                    embeddings = image_outputs
                all_image_embeddings.append(embeddings.cpu())
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
    loader=SiglipModelWrapper,
    name="google/siglip-so400m-patch14-224",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="d04cf29fca7b6374f74d8bea1969314492266b5e",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=877360306,
    n_embedding_parameters=36864000,
    memory_usage_mb=3347,
    max_tokens=16,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip-so400m-patch14-224",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip_so400m_patch14_384 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip-so400m-patch14-384",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=877960498,
    n_embedding_parameters=36864000,
    memory_usage_mb=3349,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip-so400m-patch14-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip_so400m_patch16_256_i18n = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip-so400m-patch16-256-i18n",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="365d321c0cfdea96bc28e3a29787a11a062681a1",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=1128758962,
    n_embedding_parameters=288000000,
    memory_usage_mb=4306,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip-so400m-patch16-256-i18n",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip_base_patch16_256_multilingual = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip-base-patch16-256-multilingual",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="8952a4eafcde3cb7ab46b1dd629b33f8784ca9c6",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=370626050,
    n_embedding_parameters=192000000,
    memory_usage_mb=1414,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip-base-patch16-256-multilingual",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip_base_patch16_256 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip-base-patch16-256",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="b078df89e446d623010d890864d4207fe6399f61",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=203202050,
    n_embedding_parameters=24576000,
    memory_usage_mb=775,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip-base-patch16-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip_base_patch16_512 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip-base-patch16-512",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="753a949581523b60257d93e18391e8c27f72eb22",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=203791874,
    n_embedding_parameters=24576000,
    memory_usage_mb=777,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip-base-patch16-512",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip_base_patch16_384 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip-base-patch16-384",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="41aec1c83b32e0a6fca20ad88ba058aa5b5ea394",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=203447810,
    n_embedding_parameters=24576000,
    memory_usage_mb=776,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip-base-patch16-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip_base_patch16_224 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip-base-patch16-224",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=203155970,
    n_embedding_parameters=24576000,
    memory_usage_mb=775,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip-base-patch16-224",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip_large_patch16_256 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip-large-patch16-256",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="d0da9f876e7d66b4e250cd2450c3ba2ce735e447",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=652150786,
    n_embedding_parameters=32768000,
    memory_usage_mb=2488,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip-large-patch16-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip_large_patch16_384 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip-large-patch16-384",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="ce005573a40965dfd21fd937fbdeeebf2439fc35",
    release_date="2024-01-08",
    modalities=["image", "text"],
    n_parameters=652478466,
    n_embedding_parameters=32768000,
    memory_usage_mb=2489,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip-large-patch16-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

SIGLIP2_CITATION = """@misc{tschannen2025siglip2multilingualvisionlanguage,
      title={SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features},
      author={Michael Tschannen and Alexey Gritsenko and Xiao Wang and Muhammad Ferjad Naeem and Ibrahim Alabdulmohsin and Nikhil Parthasarathy and Talfan Evans and Lucas Beyer and Ye Xia and Basil Mustafa and Olivier Henaff and Jeremiah Harmsen and Andreas Steiner and Xiaohua Zhai},
      year={2025},
      eprint={2502.14786},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}"""

siglip2_base_patch32_256 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-base-patch32-256",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="94dffa8cb1179de3e03f091dbc3917e5d5a9ae84",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=376856066,
    n_embedding_parameters=196608000,
    memory_usage_mb=1438,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-base-patch32-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_base_patch16_224 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-base-patch16-224",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=375187970,
    n_embedding_parameters=196608000,
    memory_usage_mb=1431,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-base-patch16-224",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_base_patch16_256 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-base-patch16-256",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=375234050,
    n_embedding_parameters=196608000,
    memory_usage_mb=1431,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-base-patch16-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_base_patch16_384 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-base-patch16-384",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="f775b65a79762255128c981547af89addcfe0f88",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=375479810,
    n_embedding_parameters=196608000,
    memory_usage_mb=1432,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-base-patch16-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_base_patch16_512 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-base-patch16-512",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="a89f5c5093f902bf39d3cd4d81d2c09867f0724b",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=375823874,
    n_embedding_parameters=196608000,
    memory_usage_mb=1434,
    max_tokens=64,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-base-patch16-512",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_large_patch16_256 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-large-patch16-256",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="787800c8990e6f058423089178e718139608408c",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=881526786,
    n_embedding_parameters=262144000,
    memory_usage_mb=3363,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-large-patch16-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_large_patch16_384 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-large-patch16-384",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="1b426889ea62b5a72bf9839009a1b184bfc9c178",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=881854466,
    n_embedding_parameters=262144000,
    memory_usage_mb=3364,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-large-patch16-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_large_patch16_512 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-large-patch16-512",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="49488218e80259885f3be61d7a9455faf833b7a8",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=882313218,
    n_embedding_parameters=262144000,
    memory_usage_mb=3366,
    max_tokens=64,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-large-patch16-512",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_so400m_patch14_224 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-so400m-patch14-224",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="78e403963a4f6a3640d07803284752326fdf4edf",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=1135463602,
    n_embedding_parameters=294912000,
    memory_usage_mb=4331,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-so400m-patch14-224",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_so400m_patch14_384 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-so400m-patch14-384",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=1136008498,
    n_embedding_parameters=294912000,
    memory_usage_mb=4334,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-so400m-patch14-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_so400m_patch16_256 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-so400m-patch16-256",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="e8708ab72d125807e45b36fb7d4e0aacbb59f379",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=1135670962,
    n_embedding_parameters=294912000,
    memory_usage_mb=4332,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-so400m-patch16-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_so400m_patch16_384 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-so400m-patch16-384",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="dd658faac399427308559e2c3ac1e99cbe43845d",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=1136039602,
    n_embedding_parameters=294912000,
    memory_usage_mb=4334,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-so400m-patch16-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_so400m_patch16_512 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-so400m-patch16-512",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="ceea1cba8130d8271436da4828633198c176a775",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=1136555698,
    n_embedding_parameters=294912000,
    memory_usage_mb=4336,
    max_tokens=64,
    embed_dim=1152,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-so400m-patch16-512",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_giant_opt_patch16_256 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-giant-opt-patch16-256",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="46d7129e1aa1527cc5a44d86fb35250df3abf0aa",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=1871393906,
    n_embedding_parameters=294912000,
    memory_usage_mb=7139,
    max_tokens=64,
    embed_dim=1536,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-giant-opt-patch16-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)

siglip2_giant_opt_patch16_384 = ModelMeta(
    loader=SiglipModelWrapper,
    name="google/siglip2-giant-opt-patch16-384",
    model_type=["dense"],
    languages=None,  # multilingual WebLI, no official language list
    revision="a713301b217d38485fb2204c808367d10bc3cc40",
    release_date="2025-02-17",
    modalities=["image", "text"],
    n_parameters=1871885426,
    n_embedding_parameters=294912000,
    memory_usage_mb=7141,
    max_tokens=64,
    embed_dim=1536,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/google-research/big_vision",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/google/siglip2-giant-opt-patch16-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=siglip_training_datasets,
    citation=SIGLIP2_CITATION,
    extra_requirements_groups=[
        "siglip",
        "image",
    ],
)
