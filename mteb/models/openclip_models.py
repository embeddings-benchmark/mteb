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


def openclip_loader(**kwargs):
    model_name = kwargs.get("model_name", "CLIP-ViT")
    requires_package(
        openclip_loader,
        "open_clip_torch",
        model_name,
        "pip install 'mteb[open_clip_torch]'",
    )
    import open_clip

    class OpenCLIPWrapper:
        def __init__(
            self,
            model_name: str = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
            requires_image_dependencies()

            self.model_name = model_name
            self.device = device
            self.model, _, self.img_preprocess = open_clip.create_model_and_transforms(
                f"hf-hub:{model_name}", device=device
            )
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_name}")

        def encode(  # type: ignore
            self,
            sentences: list[str],
            *,
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
            texts: list[str] | None = None,
            images: list[Image.Image] | DataLoader | None = None,
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

    return OpenCLIPWrapper(**kwargs)


CLIP_ViT_L_14_DataComp_XL_s13B_b90K = ModelMeta(
    loader=partial(
        openclip_loader,
        model_name="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    ),
    name="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    languages=["eng_Latn"],
    revision="84c9828e63dc9a9351d1fe637c346d4c1c4db341",
    release_date="2023-04-26",
    modalities=["image", "text"],
    n_parameters=428_000_000,
    memory_usage_mb=1633,
    max_tokens=77,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://huggingface.co/datasets/mlfoundations/datacomp_1b",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # DataComp-1B
    },
)

CLIP_ViT_B_32_DataComp_XL_s13B_b90K = ModelMeta(
    loader=partial(
        openclip_loader,
        model_name="laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    ),
    name="laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    languages=["eng_Latn"],
    revision="f0e2ffa09cbadab3db6a261ec1ec56407ce42912",
    release_date="2023-04-26",
    modalities=["image", "text"],
    n_parameters=151_000_000,
    memory_usage_mb=576,
    max_tokens=77,
    embed_dim=512,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://huggingface.co/datasets/mlfoundations/datacomp_1b",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # DataComp-1B
    },
)

CLIP_ViT_B_16_DataComp_XL_s13B_b90K = ModelMeta(
    loader=partial(
        openclip_loader,
        model_name="laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    ),
    name="laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    languages=["eng_Latn"],
    revision="d110532e8d4ff91c574ee60a342323f28468b287",
    release_date="2023-04-26",
    modalities=["image", "text"],
    n_parameters=150_000_000,
    memory_usage_mb=572,
    max_tokens=77,
    embed_dim=512,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://huggingface.co/datasets/mlfoundations/datacomp_1b",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # DataComp-1B
    },
)

CLIP_ViT_bigG_14_laion2B_39B_b160k = ModelMeta(
    loader=partial(
        openclip_loader,
        model_name="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    ),
    name="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    languages=["eng_Latn"],
    revision="bc7788f151930d91b58474715fdce5524ad9a189",
    release_date="2023-01-23",
    modalities=["image", "text"],
    n_parameters=2_540_000_000,
    memory_usage_mb=9689,
    max_tokens=77,
    embed_dim=1280,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://laion.ai/blog/laion-5b/",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # 2 Billion sample English subset of LAION-5B
    },
)

CLIP_ViT_g_14_laion2B_s34B_b88K = ModelMeta(
    loader=partial(
        openclip_loader,
        model_name="laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
    ),
    name="laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
    languages=["eng_Latn"],
    revision="15efd0f6ac0c40c0f9da7becca03c974d7012604",
    release_date="2023-03-06",
    modalities=["image", "text"],
    n_parameters=1_367_000_000,
    memory_usage_mb=5215,
    max_tokens=77,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://laion.ai/blog/laion-5b/",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # 2 Billion sample English subset of LAION-5B
    },
)

CLIP_ViT_H_14_laion2B_s32B_b79K = ModelMeta(
    loader=partial(
        openclip_loader,
        model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    ),
    name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    languages=["eng_Latn"],
    revision="de081ac0a0ca8dc9d1533eed1ae884bb8ae1404b",
    release_date="2022-09-15",
    modalities=["image", "text"],
    n_parameters=986_000_000,
    memory_usage_mb=3762,
    max_tokens=77,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://laion.ai/blog/laion-5b/",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # 2 Billion sample English subset of LAION-5B
    },
)

CLIP_ViT_L_14_laion2B_s32B_b82K = ModelMeta(
    loader=partial(
        openclip_loader,
        model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    ),
    name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    languages=["eng_Latn"],
    revision="1627032197142fbe2a7cfec626f4ced3ae60d07a",
    release_date="2022-09-15",
    modalities=["image", "text"],
    n_parameters=428_000_000,
    memory_usage_mb=1631,
    max_tokens=77,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://laion.ai/blog/laion-5b/",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # 2 Billion sample English subset of LAION-5B
    },
)

CLIP_ViT_B_32_laion2B_s34B_b79K = ModelMeta(
    loader=partial(
        openclip_loader,
        model_name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    ),
    name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    languages=["eng_Latn"],
    revision="08f73555f1b2fb7c82058aebbd492887a94968ef",
    release_date="2022-09-15",
    modalities=["image", "text"],
    n_parameters=151_000_000,
    memory_usage_mb=577,
    max_tokens=77,
    embed_dim=512,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/mlfoundations/open_clip",
    public_training_data="https://laion.ai/blog/laion-5b/",
    framework=["PyTorch"],
    reference="https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets={
        # 2 Billion sample English subset of LAION-5B
    },
)
