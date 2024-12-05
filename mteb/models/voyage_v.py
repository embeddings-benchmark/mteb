from __future__ import annotations

import os
from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import mteb
from mteb.model_meta import ModelMeta

api_key = os.getenv("VOYAGE_API_KEY")
tensor_to_image = transforms.Compose([transforms.ToPILImage()])


def voyage_v_loader(**kwargs):
    try:
        import voyageai
    except ImportError:
        raise ImportError("To use cohere models, please run `pip install -U voyageai`.")

    class VoyageMultiModalModelWrapper:
        def __init__(
            self,
            model_name: str,
            **kwargs: Any,
        ):
            self.model_name = model_name
            self.vo = voyageai.Client()

        def get_text_embeddings(
            self, texts: list[str], batch_size: int = 32, input_type=None
        ):
            all_text_embeddings = []

            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                batch_texts = [[text] for text in batch_texts]
                all_text_embeddings += torch.tensor(
                    self.vo.multimodal_embed(
                        batch_texts, model=self.model_name, input_type=input_type
                    ).embeddings
                )
            all_text_embeddings = torch.vstack(all_text_embeddings)
            return all_text_embeddings

        def get_image_embeddings(
            self,
            images: list[Image.Image] | DataLoader,
            batch_size: int = 32,
            input_type=None,
        ):
            all_image_embeddings = []

            if isinstance(images, DataLoader):
                for index, batch in enumerate(tqdm(images)):
                    if index == 0:
                        assert len(batch) == batch_size
                    batch_images = [[tensor_to_image(image)] for image in batch]
                    all_image_embeddings += torch.tensor(
                        self.vo.multimodal_embed(
                            batch_images, model=self.model_name, input_type=input_type
                        ).embeddings
                    )
                    # time.sleep(1.5)
            else:
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    batch_images = [[image] for image in batch_images]
                    all_image_embeddings += torch.tensor(
                        self.vo.multimodal_embed(
                            batch_images, model=self.model_name, input_type=input_type
                        ).embeddings
                    )
                    # time.sleep(1.5)
            all_image_embeddings = torch.vstack(all_image_embeddings)
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
            batch_size: int = 32,
            input_type=None,
        ):
            if texts is None and images is None:
                raise ValueError("Either texts or images must be provided")

            text_embeddings = None
            image_embeddings = None

            interleaved_embeddings = []
            if texts is not None and images is not None:
                # print("encoding interleaved inputs")
                if isinstance(images, DataLoader):
                    for index, batch in tqdm(enumerate(images)):
                        if index == 0:
                            assert len(batch) == batch_size
                        batch_images = [tensor_to_image(image) for image in batch]
                        batch_texts = texts[
                            index * batch_size : (index + 1) * batch_size
                        ]
                        interleaved_inputs = [
                            [text, image]
                            for image, text in zip(batch_images, batch_texts)
                        ]
                        interleaved_embeddings += torch.tensor(
                            self.vo.multimodal_embed(
                                interleaved_inputs,
                                model=self.model_name,
                                input_type=input_type,
                            ).embeddings
                        )
                        # time.sleep(1.5)
                else:
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_images = images[i : i + batch_size]
                        batch_texts = texts[i : i + batch_size]
                        interleaved_inputs = [
                            [text, image]
                            for image, text in zip(batch_images, batch_texts)
                        ]
                        interleaved_embeddings += torch.tensor(
                            self.vo.multimodal_embed(
                                interleaved_inputs,
                                model=self.model_name,
                                input_type=input_type,
                            ).embeddings
                        )
                        # time.sleep(1.5)
                interleaved_embeddings = torch.vstack(interleaved_embeddings)
                return interleaved_embeddings

            elif texts is not None:
                # print("encoding texts only")
                text_embeddings = self.get_text_embeddings(texts, batch_size)

            elif images is not None:
                # print("encoding images only")
                image_embeddings = self.get_image_embeddings(images, batch_size)

            if text_embeddings is not None:
                return text_embeddings
            elif image_embeddings is not None:
                return image_embeddings

    return VoyageMultiModalModelWrapper(**kwargs)


cohere_mult_3 = ModelMeta(
    loader=partial(voyage_v_loader, model_name="voyage-multimodal-3"),
    name="voyage-multimodal-3",
    languages=[],  # Unknown
    open_source=False,
    revision="1",
    release_date="2024-11-10",
    n_parameters=None,
    memory_usage=None,
    max_tokens=None,
    embed_dim=1024,
    license=None,
    similarity_fn_name="cosine",
    framework=[],
)

if __name__ == "__main__":
    mdl = mteb.get_model(cohere_mult_3.name, cohere_mult_3.revision)
    emb = mdl.encode(["Hello, world!"])
