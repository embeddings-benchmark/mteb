from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import requires_image_dependencies


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

        def encode(  # type: ignore
            self,
            sentences: list[str],
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            return self.get_text_embeddings(texts=sentences, batch_size=batch_size)

        def get_text_embeddings(
            self,
            texts: list[str],
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            all_text_embeddings = []

            with torch.no_grad(), torch.cuda.amp.autocast():
                for i in tqdm(range(0, len(texts), batch_size)):
                    batch_texts = texts[i : i + batch_size]
                    inputs = self.tokenizer(batch_texts)
                    text_outputs = self.model.encode_text(inputs.to(self.device))
                    all_text_embeddings.append(text_outputs.cpu())

            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
            return all_text_embeddings

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
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for batch in tqdm(images):
                        # import pdb; pdb.set_trace()
                        inputs = torch.vstack(
                            [
                                self.img_preprocess(F.to_pil_image(b)).unsqueeze(0)
                                for b in batch
                            ]
                        )
                        image_outputs = self.model.encode_image(inputs.to(self.device))
                        all_image_embeddings.append(image_outputs.cpu())
            else:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_images = images[i : i + batch_size]
                        inputs = torch.vstack(
                            [self.img_preprocess(b).unsqueeze(0) for b in batch_images]
                        )
                        image_outputs = self.model.encode_image(inputs.to(self.device))
                        all_image_embeddings.append(image_outputs.cpu())

            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            return all_image_embeddings

        def calculate_probs(self, text_embeddings, image_embeddings):
            text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            )
            image_embeddings = image_embeddings / image_embeddings.norm(
                dim=-1, keepdim=True
            )
            logits = torch.matmul(image_embeddings, text_embeddings.T)
            probs = (logits * 100).softmax(dim=-1)
            return probs

        def get_fused_embeddings(
            self,
            texts: list[str] = None,
            images: list[Image.Image] | DataLoader = None,
            fusion_mode="sum",
            **kwargs: Any,
        ):
            if texts is None and images is None:
                raise ValueError("Either texts or images must be provided")

            text_embeddings = None
            image_embeddings = None

            if texts is not None:
                text_embeddings = self.get_text_embeddings(texts, **kwargs)

            if images is not None:
                image_embeddings = self.get_image_embeddings(images, **kwargs)

            if text_embeddings is not None and image_embeddings is not None:
                if len(text_embeddings) != len(image_embeddings):
                    raise ValueError(
                        "The number of texts and images must have the same length"
                    )
                if fusion_mode == "sum":
                    fused_embeddings = text_embeddings + image_embeddings
                else:
                    # to do: add other fusion mode
                    raise ValueError(
                        f"fusion mode {fusion_mode} hasn't been implemented"
                    )
                return fused_embeddings
            elif text_embeddings is not None:
                return text_embeddings
            elif image_embeddings is not None:
                return image_embeddings

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
    loader=partial(
        evaclip_loader,
        model_name="EVA02-CLIP-B-16",
    ),
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
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=training_datasets,
)

EVA02_CLIP_L_14 = ModelMeta(
    loader=partial(
        evaclip_loader,
        model_name="EVA02-CLIP-L-14",
    ),
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
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=training_datasets,
)

EVA02_CLIP_bigE_14 = ModelMeta(
    loader=partial(
        evaclip_loader,
        model_name="EVA02-CLIP-bigE-14",
    ),
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
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=laion_2b,
)


EVA02_CLIP_bigE_14_plus = ModelMeta(
    loader=partial(
        evaclip_loader,
        model_name="EVA02-CLIP-bigE-14-plus",
    ),
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
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=laion_2b,
)
