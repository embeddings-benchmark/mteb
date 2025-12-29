from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

BLIP2_CITATION = """@inproceedings{li2023blip2,
    title={{BLIP-2:} Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
    author={Junnan Li and Dongxu Li and Silvio Savarese and Steven Hoi},
    year={2023},
    booktitle={ICML},
}"""


def blip2_loader(model_name, **kwargs):
    requires_package(
        blip2_loader, "salesforce-lavis", model_name, "pip install 'mteb[blip2]'"
    )
    from lavis.models.blip2_models.blip2_image_text_matching import (
        Blip2ITM,
    )

    class BLIP2Model(AbsEncoder):
        def __init__(
            self,
            model_name: str,
            revision: str,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs: Any,
        ):
            from transformers import Blip2Processor

            self.model_name = model_name
            self.device = device
            model_type = "coco" if "coco" in model_name else "pretrain"
            self.model = (
                Blip2ITM.from_pretrained(model_type, revision=revision)
                .to(self.device)
                .float()
            )
            self.processor = Blip2Processor.from_pretrained(
                model_name, revision=revision
            )

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
            # TODO: find out if BLIP has a prescribed way of fusing text and image embeddings
            text_embeddings = None
            image_embeddings = None

            if "text" in inputs.dataset.features and "image" in inputs.dataset.features:
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
                fused_embeddings = text_embeddings + image_embeddings
                return fused_embeddings
            elif text_embeddings is not None:
                return text_embeddings
            elif image_embeddings is not None:
                return image_embeddings
            raise ValueError

    return BLIP2Model(model_name, **kwargs)


blip2_training_datasets = set(
    # COCO
    # CC3M+CC12M+SBU
    # LAION400M
)

blip2_opt_2_7b = ModelMeta(
    loader=blip2_loader,
    name="Salesforce/blip2-opt-2.7b",
    model_type=["dense"],
    languages=["eng-Latn"],
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
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=blip2_training_datasets,
    citation=BLIP2_CITATION,
)

blip2_opt_6_7b_coco = ModelMeta(
    loader=blip2_loader,
    name="Salesforce/blip2-opt-6.7b-coco",
    model_type=["dense"],
    languages=["eng-Latn"],
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
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=blip2_training_datasets,
    citation=BLIP2_CITATION,
)
