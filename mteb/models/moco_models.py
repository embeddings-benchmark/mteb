from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_image_dependencies, requires_package


def mocov3_loader(**kwargs):
    model_name = kwargs.get("model_name", "MOCOv3")
    requires_package(mocov3_loader, "timm", model_name, "pip install 'mteb[timm]'")
    import timm

    class MOCOv3Wrapper:
        """A wrapper class for MOCOv3 models that supports image encoding.
        Text encoding and text-image fusion are not supported.
        """

        def __init__(
            self,
            model_name: str = "nyu-visionx/moco-v3-vit-b",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
            requires_image_dependencies()

            self.model_name = model_name
            self.device = device
            name = "vit_base_patch16_224"
            if "vit-l" in model_name:
                name = "vit_large_patch16_224"
            model = timm.create_model(
                name,
                pretrained=True,
                num_classes=0,
                pretrained_cfg_overlay={"hf_hub_id": model_name},
            )

            self.model = model.eval()

            # get model specific transforms (normalization, resize)
            data_config = timm.data.resolve_model_data_config(self.model)
            self.processor = timm.data.create_transform(
                **data_config, is_training=False
            )

        @staticmethod
        def get_text_embeddings(
            texts: list[str],
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            raise ValueError("MOCO models only support image encoding.")

        def get_image_embeddings(
            self,
            images: list[Image.Image] | DataLoader,
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            import torchvision.transforms.functional as F

            all_image_embeddings = []

            if isinstance(images, DataLoader):
                with torch.no_grad():
                    for batch in tqdm(images):
                        inputs = torch.vstack(
                            [
                                self.processor(F.to_pil_image(b.to("cpu"))).unsqueeze(0)
                                for b in batch
                            ]
                        )
                        output = self.model(
                            inputs
                        )  # output is (batch_size, num_features) shaped tensor
                        all_image_embeddings.append(output)
            else:
                with torch.no_grad():
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_images = images[i : i + batch_size]
                        inputs = torch.vstack(
                            [self.processor(b).unsqueeze(0) for b in batch_images]
                        )
                        output = self.model(
                            inputs
                        )  # output is (batch_size, num_features) shaped tensor
                        all_image_embeddings.append(output)

            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            return all_image_embeddings

        @staticmethod
        def calculate_probs(text_embeddings, image_embeddings):
            raise ValueError("MOCO models only support image encoding.")

        def get_fused_embeddings(
            self,
            texts: list[str] | None = None,
            images: list[Image.Image] | DataLoader | None = None,
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            fusion_mode="sum",
            **kwargs: Any,
        ):
            if texts is None and images is None:
                raise ValueError("images must be provided for MOCO models")

            text_embeddings = None
            image_embeddings = None

            if texts is not None:
                text_embeddings = self.get_text_embeddings(texts, batch_size)

            if images is not None:
                image_embeddings = self.get_image_embeddings(images, batch_size)

            if text_embeddings is not None and image_embeddings is not None:
                raise ValueError("MOCO models only support image encoding.")
            elif text_embeddings is not None:
                return text_embeddings
            elif image_embeddings is not None:
                return image_embeddings

    return MOCOv3Wrapper(**kwargs)


mocov3_training_datasets = {
    # imagenet
}

mocov3_vit_base = ModelMeta(
    loader=partial(
        mocov3_loader,
        model_name="nyu-visionx/moco-v3-vit-b",
    ),
    name="nyu-visionx/moco-v3-vit-b",
    languages=["eng_Latn"],
    revision="7d091cd70772c5c0ecf7f00b5f12ca609a99d69d",
    release_date="2024-06-03",
    modalities=["image"],
    n_parameters=86_600_000,
    memory_usage_mb=330,
    max_tokens=None,
    embed_dim=768,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/moco-v3",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://github.com/facebookresearch/moco-v3",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=mocov3_training_datasets,
)

mocov3_vit_large = ModelMeta(
    loader=partial(
        mocov3_loader,
        model_name="nyu-visionx/moco-v3-vit-l",
    ),
    name="nyu-visionx/moco-v3-vit-l",
    languages=["eng_Latn"],
    revision="7bf75358d616f39b9716148bf4e3425f3bd35b47",
    release_date="2024-06-03",
    modalities=["image"],
    n_parameters=304_000_000,
    memory_usage_mb=1161,
    max_tokens=None,
    embed_dim=1024,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/moco-v3",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://github.com/facebookresearch/moco-v3",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=mocov3_training_datasets,
)
