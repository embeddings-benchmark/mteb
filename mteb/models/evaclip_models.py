from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.abstasks import TaskMetadata
from mteb.model_meta import ModelMeta, ScoringFunction
from mteb.requires_package import requires_image_dependencies
from mteb.types import Array, BatchedInput, PromptType


def evaclip_loader(**kwargs):
    try:
        import os
        import sys

        sys.path.insert(0, os.path.join(os.getcwd(), "EVA/EVA-CLIP/rei"))

        from eva_clip import create_model_and_transforms, get_tokenizer
    except ImportError:
        # https://github.com/baaivision/EVA/tree/master/EVA-CLIP#setup
        raise ImportError(
            "Please run `git clone git@github.com:baaivision/EVA.git`,"
            "`pip install ninja timm`"
            "`pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers`"
            "`git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./`"
        )

    class EvaCLIPWrapper:
        def __init__(
            self,
            model_name: str = "EVA02-CLIP-B-16",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
            requires_image_dependencies()

            self.model_name = model_name
            self.device = device
            pretrained = "eva_clip"  # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
            self.model, _, self.img_preprocess = create_model_and_transforms(
                model_name, pretrained, force_custom_clip=True, device=device
            )
            self.model.eval()
            self.tokenizer = get_tokenizer(model_name)

        def get_text_embeddings(
            self,
            texts: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_text_embeddings = []

            with torch.no_grad(), torch.cuda.amp.autocast():
                for batch in tqdm(
                    texts, disable=not show_progress_bar, desc="Text Encoding"
                ):
                    inputs = self.tokenizer(batch["text"])
                    text_outputs = self.model.encode_text(inputs.to(self.device))
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

            with torch.no_grad(), torch.cuda.amp.autocast():
                for batch in tqdm(
                    images, disable=not show_progress_bar, desc="Image Encoding"
                ):
                    inputs = torch.vstack(
                        [self.img_preprocess(b).unsqueeze(0) for b in batch["image"]]
                    )
                    image_outputs = self.model.encode_image(inputs.to(self.device))
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

    return EvaCLIPWrapper(**kwargs)


training_code = "https://github.com/baaivision/EVA/tree/master/EVA-CLIP"
training_datasets = {
    # COYO-700M, random sample 400M. https://github.com/kakaobrain/coyo-dataset
    # LAION-2B, random sample 1.6B. https://laion.ai/blog/laion-5b/
}
laion_2b = {
    # LAION-2B
}

EVA02_CLIP_B_16 = ModelMeta(
    loader=evaclip_loader,
    name="QuanSun/EVA02-CLIP-B-16",
    languages=["eng_Latn"],
    revision="11afd202f2ae80869d6cef18b1ec775e79bd8d12",
    release_date="2023-04-26",
    modalities=["image", "text"],
    n_parameters=149_000_000,
    memory_usage_mb=568,
    max_tokens=77,
    embed_dim=512,
    license="mit",
    open_weights=True,
    public_training_code=training_code,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/QuanSun/EVA-CLIP",
    similarity_fn_name=ScoringFunction.VISION,
    use_instructions=False,
    training_datasets=training_datasets,
)

EVA02_CLIP_L_14 = ModelMeta(
    loader=evaclip_loader,
    name="QuanSun/EVA02-CLIP-L-14",
    languages=["eng_Latn"],
    revision="11afd202f2ae80869d6cef18b1ec775e79bd8d12",
    release_date="2023-04-26",
    modalities=["image", "text"],
    n_parameters=428_000_000,
    memory_usage_mb=1633,
    max_tokens=77,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code=training_code,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/QuanSun/EVA-CLIP",
    similarity_fn_name=ScoringFunction.VISION,
    use_instructions=False,
    training_datasets=training_datasets,
)

EVA02_CLIP_bigE_14 = ModelMeta(
    loader=evaclip_loader,
    name="QuanSun/EVA02-CLIP-bigE-14",
    languages=["eng_Latn"],
    revision="11afd202f2ae80869d6cef18b1ec775e79bd8d12",
    release_date="2023-04-26",
    modalities=["image", "text"],
    n_parameters=4_700_000_000,
    memory_usage_mb=17929,
    max_tokens=77,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code=training_code,
    public_training_data="https://laion.ai/blog/laion-5b/",
    framework=["PyTorch"],
    reference="https://huggingface.co/QuanSun/EVA-CLIP",
    similarity_fn_name=ScoringFunction.VISION,
    use_instructions=False,
    training_datasets=laion_2b,
)


EVA02_CLIP_bigE_14_plus = ModelMeta(
    loader=evaclip_loader,
    name="QuanSun/EVA02-CLIP-bigE-14-plus",
    languages=["eng_Latn"],
    revision="11afd202f2ae80869d6cef18b1ec775e79bd8d12",
    release_date="2023-04-26",
    modalities=["image", "text"],
    n_parameters=5_000_000_000,
    memory_usage_mb=19073,
    max_tokens=77,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code=training_code,
    public_training_data="https://laion.ai/blog/laion-5b/",
    framework=["PyTorch"],
    reference="https://huggingface.co/QuanSun/EVA-CLIP",
    similarity_fn_name=ScoringFunction.VISION,
    use_instructions=False,
    training_datasets=laion_2b,
)
