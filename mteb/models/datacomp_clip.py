from __future__ import annotations

from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb.model_meta import ModelMeta


def datacomp_clip_loader(**kwargs):
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "Please install `pip install open_clip_torch` to use DataComp CLIP models."
        )

    class DataCLIPModelWrapper:
        def __init__(
            self,
            model_name: str = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
            self.model_name = model_name
            self.device = device
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                model_name=f"hf-hub:{model_name}",
                device=self.device,
            )
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_name}")

        def preprocess(
            self,
            texts: list[str] = [],
            images: list[Image.Image] = [],
        ):
            if len(texts):
                return self.tokenizer(texts)
            return self.processor(images)

        def get_text_embeddings(self, texts: list[str], batch_size: int = 32):
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
            self, images: list[Image.Image] | DataLoader, batch_size: int = 32
        ):
            all_image_embeddings = []

            if isinstance(images, DataLoader):
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for batch in tqdm(images):
                        inputs = [self.processor(b) for b in batch]
                        batch_images = torch.stack(inputs, dim=0)
                        image_outputs = self.model.encode_image(inputs.to(self.device))
                        all_image_embeddings.append(image_outputs.cpu())
            else:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_images = images[i : i + batch_size]
                        inputs = [self.processor(b) for b in batch_images]
                        inputs = torch.stack(inputs, dim=0)
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
            batch_size: int = 32,
        ):
            if texts is None and images is None:
                raise ValueError("Either texts or images must be provided")

            text_embeddings = None
            image_embeddings = None

            if texts is not None:
                text_embeddings = self.get_text_embeddings(texts, batch_size)

            if images is not None:
                image_embeddings = self.get_image_embeddings(images, batch_size)

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

    return DataCLIPModelWrapper(**kwargs)


datacomp_clip_vit_large_patch14 = ModelMeta(
    loader=partial(
        datacomp_clip_loader,
        model_name="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    ),
    name="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    languages=["eng_Latn"],
    open_source=True,
    revision="84c9828e63dc9a9351d1fe637c346d4c1c4db341",
    release_date="2023-04-26",
)

datacomp_clip_vit_base_patch32 = ModelMeta(
    loader=partial(
        datacomp_clip_loader,
        model_name="laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    ),
    name="laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    languages=["eng_Latn"],
    open_source=True,
    revision="f0e2ffa09cbadab3db6a261ec1ec56407ce42912",
    release_date="2023-04-26",
)

datacomp_clip_vit_base_patch16 = ModelMeta(
    loader=partial(
        datacomp_clip_loader,
        model_name="laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    ),
    name="laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    languages=["eng_Latn"],
    open_source=True,
    revision="d110532e8d4ff91c574ee60a342323f28468b287",
    release_date="2023-04-26",
)


if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(
        datacomp_clip_vit_large_patch14.name, datacomp_clip_vit_large_patch14.revision
    )
    test_txt = "Hello, world!"
    emb = mdl.get_text_embeddings([test_txt])

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", device=mdl.device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(
        "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    )
    text = tokenizer([test_txt])

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text.to(mdl.device))
        assert torch.allclose(text_features.cpu(), emb)

    #### images
    # Download the image by running `wget https://github.com/mlfoundations/open_clip/blob/main/docs/CLIP.png`
    from PIL import Image

    im = Image.open("CLIP.png")
    image = preprocess(im).unsqueeze(0)
    img_emb = mdl.get_image_embeddings([im])

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image.to(mdl.device))
        assert torch.allclose(image_features.cpu(), img_emb)
