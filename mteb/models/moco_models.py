from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.encoder_interface import BatchedInput, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


def mocov3_loader(**kwargs):
    try:
        import timm
    except ImportError:
        raise ImportError("Please install `pip install timm` to use MOCOv3 models.")

    class MOCOv3Wrapper(Wrapper):
        """A wrapper class for MOCOv3 models that supports image encoding.
        Text encoding and text-image fusion are not supported.
        """

        def __init__(
            self,
            model_name: str = "nyu-visionx/moco-v3-vit-b",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
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
            self,
            texts: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            raise ValueError("MOCO models only support image encoding.")

        def get_image_embeddings(
            self,
            images: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_image_embeddings = []

            import torchvision.transforms.functional as F

            with torch.no_grad():
                for batch in tqdm(
                    images, disable=not show_progress_bar, desc="Image Encoding"
                ):
                    inputs = torch.vstack(
                        [
                            self.processor(F.to_pil_image(b.to("cpu"))).unsqueeze(0)
                            for b in batch["image"]
                        ]
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

        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *,
            task_name: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> np.ndarray | torch.Tensor:
            0 / 0
            if "text" in inputs.dataset.features:
                raise ValueError(
                    "MOCO models only support image encoding. Text encoding is not supported."
                )
            if "image" in inputs.dataset.features:
                return self.get_image_embeddings(inputs, **kwargs)

    return MOCOv3Wrapper(**kwargs)


mocov3_training_datasets = {
    # imagenet
}

mocov3_vit_base = ModelMeta(
    loader=mocov3_loader,  # type: ignore
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
    loader=mocov3_loader,  # type: ignore
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
