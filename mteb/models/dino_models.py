from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta


class DINOModelWrapper:
    """A wrapper class for DINO models that supports image encoding.
    Text encoding and text-image fusion are not supported.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    @staticmethod
    def get_text_embeddings(
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        raise ValueError("DINO models only support image encoding.")

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        pooling="cls",
        **kwargs: Any,
    ):
        all_image_embeddings = []

        if isinstance(images, DataLoader):
            with torch.no_grad():
                for batch in tqdm(images):
                    inputs = self.processor(images=batch, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model(**inputs)
                    features = image_outputs.last_hidden_state
                    if pooling == "cls":
                        features = features[:, 0, :]  # TODO: confirm best practice
                    elif pooling == "mean":
                        features = features.mean(dim=1)
                    else:
                        raise ValueError(
                            "Pooling methods not implemented. Use cls or mean."
                        )
                    all_image_embeddings.append(features.cpu())
        else:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    inputs = self.processor(images=batch_images, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_outputs = self.model(**inputs)
                    features = image_outputs.last_hidden_state
                    if pooling == "cls":
                        features = features[:, 0, :]
                    elif pooling == "mean":
                        features = features.mean(dim=1)
                    else:
                        raise ValueError(
                            "Pooling methods not implemented. Use cls or mean."
                        )
                    all_image_embeddings.append(features.cpu())

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings

    @staticmethod
    def calculate_probs(text_embeddings, image_embeddings):
        raise ValueError("DINO models only support image encoding.")

    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list[Image.Image] | DataLoader = None,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        if texts is None and images is None:
            raise ValueError("images must be provided for DINO models")

        text_embeddings = None
        image_embeddings = None

        if texts is not None:
            text_embeddings = self.get_text_embeddings(texts, **kwargs)

        if images is not None:
            image_embeddings = self.get_image_embeddings(images, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            raise ValueError("DINO models only support image encoding.")
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings


dinov2_training_datasets = {
    # LVD-142M
    #  ImageNet-22k
}


dinov2_small = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/dinov2-small",
    ),
    name="facebook/dinov2-small",
    languages=["eng-Latn"],
    revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
    release_date="2023-07-18",
    modalities=["image"],
    n_parameters=22_100_000,
    memory_usage_mb=84,
    max_tokens=None,
    embed_dim=384,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/dinov2",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/dinov2-small",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)

dinov2_base = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/dinov2-base",
    ),
    name="facebook/dinov2-base",
    languages=["eng-Latn"],
    revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    release_date="2023-07-18",
    modalities=["image"],
    n_parameters=86_600_000,
    memory_usage_mb=330,
    max_tokens=None,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/dinov2",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/dinov2-base",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)

dinov2_large = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/dinov2-large",
    ),
    name="facebook/dinov2-large",
    languages=["eng-Latn"],
    revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    release_date="2023-07-18",
    modalities=["image"],
    n_parameters=304_000_000,
    memory_usage_mb=1161,
    max_tokens=None,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/dinov2",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/dinov2-large",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)

dinov2_giant = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/dinov2-giant",
    ),
    name="facebook/dinov2-giant",
    languages=["eng-Latn"],
    revision="611a9d42f2335e0f921f1e313ad3c1b7178d206d",
    release_date="2023-07-18",
    modalities=["image"],
    n_parameters=1_140_000_000,
    memory_usage_mb=4335,
    max_tokens=None,
    embed_dim=1536,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/dinov2",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/dinov2-giant",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=dinov2_training_datasets,
)

webssl_dino_training_datasets = {
    # MetaCLIP 2B samples
}

webssl_dino300m_full2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino300m-full2b-224",
    ),
    name="facebook/webssl-dino300m-full2b-224",
    languages=["eng-Latn"],
    revision="8529cdb3fb75014932af3b896455fc21c386168e",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=304_000_000,
    memory_usage_mb=1158,
    max_tokens=None,
    embed_dim=1024,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino300m-full2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_dino1b_full2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino1b-full2b-224",
    ),
    name="facebook/webssl-dino1b-full2b-224",
    languages=["eng-Latn"],
    revision="d3bf033d9c8cc62ea9e73c40956642cad2ec568a",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=1_130_000_000,
    memory_usage_mb=4329,
    max_tokens=None,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino1b-full2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_dino2b_full2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino2b-full2b-224",
    ),
    name="facebook/webssl-dino2b-full2b-224",
    languages=["eng-Latn"],
    revision="cd5893e3fd2e988eb716792049b3dd53b3f1b68b",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=2_080_000_000,
    memory_usage_mb=7951,
    max_tokens=None,
    embed_dim=2688,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino2b-full2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_dino3b_full2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino3b-full2b-224",
    ),
    name="facebook/webssl-dino3b-full2b-224",
    languages=["eng-Latn"],
    revision="2d015c340b16bc47bc6557fcb4e6c83a9d4aa1d3",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=3_000_000_000,
    memory_usage_mb=11247,
    max_tokens=None,
    embed_dim=3072,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino3b-full2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_dino5b_full2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino5b-full2b-224",
    ),
    name="facebook/webssl-dino5b-full2b-224",
    languages=["eng-Latn"],
    revision="88006b18b9af369f6c611db7a64d908bde3714e0",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=5_000_000_000,
    memory_usage_mb=18838,
    max_tokens=None,
    embed_dim=3584,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino5b-full2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_dino7b_full8b_224 = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino7b-full8b-224",
    ),
    name="facebook/webssl-dino7b-full8b-224",
    languages=["eng-Latn"],
    revision="c6085463ea680043042a80c6d41db2c65e85f466",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=7_000_000_000,
    memory_usage_mb=24605,
    max_tokens=None,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino7b-full8b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_dino7b_full8b_378 = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino7b-full8b-378",
    ),
    name="facebook/webssl-dino7b-full8b-378",
    languages=["eng-Latn"],
    revision="53c8c5b43070bd2ddb3f66161140408ce832301f",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=7_000_000_000,
    memory_usage_mb=24613,
    max_tokens=None,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino7b-full8b-378",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_dino7b_full8b_518 = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino7b-full8b-518",
    ),
    name="facebook/webssl-dino7b-full8b-518",
    languages=["eng-Latn"],
    revision="aee350d2c5e3e5fdb7ee6985291d808ea5eef431",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=7_000_000_000,
    memory_usage_mb=24623,
    max_tokens=None,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino7b-full8b-518",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_dino2b_light2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino2b-light2b-224",
    ),
    name="facebook/webssl-dino2b-light2b-224",
    languages=["eng-Latn"],
    revision="633a663f304e63cc3cbec3f7f9ca2fbc94736128",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=2_000_000_000,
    memory_usage_mb=7951,
    max_tokens=None,
    embed_dim=2688,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino2b-light2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_dino2b_heavy2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino2b-heavy2b-224",
    ),
    name="facebook/webssl-dino2b-heavy2b-224",
    languages=["eng-Latn"],
    revision="9f46eb0c0129656a1ef195fde072e3765abdb7c6",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=2_000_000_000,
    memory_usage_mb=7951,
    max_tokens=None,
    embed_dim=2688,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino2b-heavy2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_dino3b_light2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino3b-light2b-224",
    ),
    name="facebook/webssl-dino3b-light2b-224",
    languages=["eng-Latn"],
    revision="4d0160f60673805431f4ad14983e712ed88be5b8",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=3_000_000_000,
    memory_usage_mb=11247,
    max_tokens=None,
    embed_dim=3072,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino3b-light2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_dino3b_heavy2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-dino3b-heavy2b-224",
    ),
    name="facebook/webssl-dino3b-heavy2b-224",
    languages=["eng-Latn"],
    revision="dd39c2910747561b332285d96c4dce0bdb240775",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=3_000_000_000,
    memory_usage_mb=11247,
    max_tokens=None,
    embed_dim=3072,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-dino3b-heavy2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_mae300m_full2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-mae300m-full2b-224",
    ),
    name="facebook/webssl-mae300m-full2b-224",
    languages=["eng-Latn"],
    revision="4655a0ac1726c206ba14d5ccb26758c62a4d03b0",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=304_000_000,
    memory_usage_mb=1161,
    max_tokens=None,
    embed_dim=1024,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-mae300m-full2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_mae700m_full2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-mae700m-full2b-224",
    ),
    name="facebook/webssl-mae700m-full2b-224",
    languages=["eng-Latn"],
    revision="c32be382e757d73a178de1ead62c27391d4b4280",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=700_000_000,
    memory_usage_mb=2412,
    max_tokens=None,
    embed_dim=1280,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-mae700m-full2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)

webssl_mae1b_full2b = ModelMeta(
    loader=partial(
        DINOModelWrapper,
        model_name="facebook/webssl-mae1b-full2b-224",
    ),
    name="facebook/webssl-mae1b-full2b-224",
    languages=["eng-Latn"],
    revision="5880aefedbad8db0f44d27358f6f08e8576f70fc",
    release_date="2025-04-24",
    modalities=["image"],
    n_parameters=1_000_000_000,
    memory_usage_mb=4337,
    max_tokens=None,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/facebook/webssl-mae1b-full2b-224",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=webssl_dino_training_datasets,
)
