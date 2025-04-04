from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Blip2Processor

from mteb.encoder_interface import BatchedInput, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package


def blip2_loader(**kwargs):
    model_name = kwargs.get("model_name", "BLIP-2")
    requires_package(
        blip2_loader, "salesforce-lavis", model_name, "pip install 'mteb[blip2]'"
    )
    from lavis.models.blip2_models.blip2_image_text_matching import (
        Blip2ITM,
    )

    class BLIP2ModelWrapper(Wrapper):
        def __init__(
            self,
            model_name: str,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
            self.model_name = model_name
            self.device = device
            model_type = "coco" if "coco" in model_name else "pretrain"
            self.model = Blip2ITM.from_pretrained(model_type).to(self.device).float()
            self.processor = Blip2Processor.from_pretrained(model_name)

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
                    text_tokens = self.model.tokenizer(
                        batch["text"],
                        padding="max_length",
                        truncation=True,
                        max_length=self.model.max_txt_len,
                        return_tensors="pt",
                    ).to(self.device)
                    text_outputs = self.model.forward_text(text_tokens)
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
                for batch in tqdm(
                    images, disable=not show_progress_bar, desc="Image Encoding"
                ):
                    inputs = self.processor(
                        images=batch["image"], return_tensors="pt", padding=True
                    )
                    image_outputs = self.model.forward_image(
                        inputs["pixel_values"].to(self.device)
                    )
                    image_outputs = image_outputs[0][:, 0, :]
                    all_image_embeddings.append(image_outputs.cpu())

            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            return all_image_embeddings

        def get_multimodal_embeddings(
            self,
            inputs: DataLoader[BatchedInput],
            show_progress_bar: bool = True,
            **kwargs: Any,
        ):
            all_multimodal_embeddings = []

            with torch.no_grad():
                # check dataloader batch size is the same as batch size
                for batch in tqdm(
                    inputs, disable=not show_progress_bar, desc="Multimodal Encoding"
                ):
                    image_inputs = self.processor(
                        images=batch["image"], return_tensors="pt", padding=True
                    )["pixel_values"].to(self.device)
                    multimodal_outputs = self.model.extract_features(
                        {"text_input": batch["text"], "image": image_inputs}
                    ).multimodal_embeds[:, 0, :]
                    all_multimodal_embeddings.append(multimodal_outputs.cpu())

            return torch.cat(all_multimodal_embeddings, dim=0)

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

        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *,
            task_name: str,
            prompt_type: PromptType | None = None,
            fusion_mode: Literal["sum", "multimodal"] = "sum",
            **kwargs: Any,
        ) -> np.ndarray | torch.Tensor:
            # TODO: find out if BLIP has a prescribed way of fusing text and image embeddings
            text_embeddings = None
            image_embeddings = None

            if (
                "text" in inputs.dataset.features
                and "image" in inputs.dataset.features
                and fusion_mode == "multimodal"
            ):
                return self.get_multimodal_embeddings(inputs)

            if "text" in inputs.dataset.features:
                text_embeddings = self.get_text_embeddings(inputs, **kwargs)
            if "image" in inputs.dataset.features:
                image_embeddings = self.get_image_embeddings(inputs, **kwargs)

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

    return BLIP2ModelWrapper(**kwargs)


blip2_training_datasets = {
    # COCO
    # CC3M+CC12M+SBU
    # LAION400M
}

blip2_opt_2_7b = ModelMeta(
    loader=blip2_loader,
    name="Salesforce/blip2-opt-2.7b",
    languages=["eng_Latn"],
    revision="51572668da0eb669e01a189dc22abe6088589a24",
    release_date="2024-03-22",
    modalities=["image", "text"],
    n_parameters=3_740_000_000,
    memory_usage_mb=14285,
    max_tokens=None,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/salesforce/LAVIS/tree/main/projects/blip2",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip2-opt-2.7b",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=blip2_training_datasets,
)

blip2_opt_6_7b_coco = ModelMeta(
    loader=blip2_loader,
    name="Salesforce/blip2-opt-6.7b-coco",
    languages=["eng_Latn"],
    revision="0d580de59320a25a4d2c386387bcef310d5f286e",
    release_date="2024-03-31",
    modalities=["image", "text"],
    n_parameters=7_750_000_000,
    memory_usage_mb=29577,
    max_tokens=None,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/salesforce/LAVIS/tree/main/projects/blip2",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/Salesforce/blip2-opt-6.7b-coco",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=blip2_training_datasets,
)
