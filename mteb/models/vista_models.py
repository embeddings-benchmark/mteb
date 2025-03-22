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


def vista_loader(**kwargs):
    try:  # a temporal fix for the dependency issues of vista models.
        from visual_bge.modeling import Visualized_BGE
    except ImportError:
        raise ImportError(
            "Please install `visual_bge`, refer to https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge#install-flagembedding."
        )

    class VisualizedBGEWrapper(Visualized_BGE):
        """Setting up VISTA

        ```
        git clone https://github.com/FlagOpen/FlagEmbedding.git
        cd FlagEmbedding/research/visual_bge
        pip install -e .
        pip install torchvision timm einops ftfy
        ```
        back to the root folder of mteb; download the vision tower for bge-base
        ```
        cd ..
        wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth?download=true
        ```
        rename it to `visualized_base_en_V1.5.pth`
        ```
        mv Visualized_base_en_v1.5.pth?download=true visualized_base_en_V1.5.pth
        ```
        download the vision tower for bge-m3
        ```
        wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_m3.pth?download=true
        ```
        rename it to `visualized_m3.pth`
        ```
        mv Visualized_m3.pth?download=true visualized_m3.pth
        ```
        """

        def __init__(
            self,
            model_name_bge: str | None = None,
            model_weight=None,
            normlized: bool = True,
            sentence_pooling_method: str = "cls",
            negatives_cross_device: bool = False,
            temperature: float = 0.02,
            from_pretrained=None,
            image_tokens_num: int | None = None,
            **kwargs: Any,
        ):
            requires_image_dependencies()
            from torchvision import transforms

            super().__init__(
                model_name_bge=model_name_bge,
                model_weight=model_weight,
                normlized=normlized,
                sentence_pooling_method=sentence_pooling_method,
                negatives_cross_device=negatives_cross_device,
                temperature=temperature,
                from_pretrained=from_pretrained,
            )
            self.image_tokens_num = image_tokens_num
            self.max_text_len_with_image = (
                self.tokenizer.model_max_length - image_tokens_num
            )
            self.transform = transforms.Compose([transforms.ToPILImage()])
            self.eval()

        def encode_text(self, texts):
            """Currently override Visualized_BGE's the original implementation
            to fix attention_mask & embedding_output dtype misalignment
            """
            input_ids = texts["input_ids"]
            attention_mask = texts["attention_mask"]

            input_shape = input_ids.size()
            device = input_ids.device

            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            head_mask = [None] * self.depth
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_shape
            )

            embedding_output = self.bge_embeddings(
                input_ids=input_ids,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=0,
            )

            # this line is missing in vista, currently override "encode_text" only to fix this.
            extended_attention_mask = extended_attention_mask.to(embedding_output.dtype)

            encoder_outputs = self.bge_encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            sequence_output = encoder_outputs[0]

            t_reps = self.sentence_embedding(
                sequence_output, texts["attention_mask"]
            )  # tensor: reps with pooling
            if self.normlized:
                t_reps = torch.nn.functional.normalize(t_reps, dim=-1)
            return t_reps.contiguous()

        def encode(
            self,
            images=None,
            texts=None,
            tensors=False,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ):
            if images is not None:
                if isinstance(images, list):
                    if not tensors:
                        images = [
                            self.preprocess_val(
                                img if isinstance(img, Image.Image) else Image.open(img)
                            )
                            for img in images
                        ]
                    else:
                        images = [
                            self.preprocess_val(self.tensor_to_image(image))
                            for image in images
                        ]
                    images = torch.stack(images)
                if texts is not None:
                    texts = self.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_text_len_with_image,
                    )
                    return self.encode_mm(images.to(self.device), texts.to(self.device))
                else:
                    return self.encode_image(images.to(self.device))
            else:
                if texts is not None:
                    texts = self.tokenizer(
                        texts, return_tensors="pt", padding=True, truncation=True
                    )
                    return self.encode_text(texts.to(self.device))
                else:
                    return None

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
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                with torch.no_grad():
                    batch_embeddings = self.encode(texts=batch_texts)
                all_text_embeddings.append(batch_embeddings.cpu())
            return torch.cat(all_text_embeddings, dim=0)

        def get_image_embeddings(
            self,
            images: list[Image.Image] | DataLoader,
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            all_image_embeddings = []

            if isinstance(images, DataLoader):
                with torch.no_grad():
                    for batch in tqdm(images):
                        batch_embeddings = self.encode(images=batch, tensors=True)
                        all_image_embeddings.append(batch_embeddings.cpu())
            else:
                with torch.no_grad():
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_images = images[i : i + batch_size]
                        batch_embeddings = self.encode(images=batch_images)
                        all_image_embeddings.append(batch_embeddings.cpu())
            return torch.cat(all_image_embeddings, dim=0)

        def get_fused_embeddings(
            self,
            texts: list[str] = None,
            images: list[Image.Image] | DataLoader = None,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            all_embeddings = []

            if isinstance(images, DataLoader):
                with torch.no_grad():
                    for index, batch_images in enumerate(tqdm(images)):
                        batch_texts = texts[
                            index * batch_size : (index + 1) * batch_size
                        ]
                        batch_embeddings = self.encode(
                            images=batch_images, texts=batch_texts, tensors=True
                        )
                        all_embeddings.append(batch_embeddings.cpu())
            else:
                assert len(texts) == len(images)
                with torch.no_grad():
                    for i in tqdm(range(0, len(texts), batch_size)):
                        batch_texts = texts[i : i + batch_size]
                        batch_images = images[i : i + batch_size]
                        batch_embeddings = self.encode(
                            images=batch_images, texts=batch_texts
                        )
                        all_embeddings.append(batch_embeddings.cpu())
            return torch.cat(all_embeddings, dim=0)

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

    return VisualizedBGEWrapper(**kwargs)


vista_training_datasets = {
    # VISTA_S2
}

visualized_bge_base = ModelMeta(
    loader=partial(
        vista_loader,
        model_name_bge="BAAI/bge-base-en-v1.5",
        model_weight="visualized_base_en_V1.5.pth",
        image_tokens_num=196,
    ),
    name="BAAI/bge-visualized-base",
    languages=["eng_Latn"],
    revision="98db10b10d22620010d06f11733346e1c98c34aa",
    release_date="2024-06-06",
    modalities=["image", "text"],
    n_parameters=196_000_000,
    memory_usage_mb=1631,
    max_tokens=512,
    embed_dim=768,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/JUNJIE99/VISTA_S2",
    framework=["PyTorch"],
    reference="https://huggingface.co/BAAI/bge-visualized",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=vista_training_datasets,
)

visualized_bge_m3 = ModelMeta(
    loader=partial(
        vista_loader,
        model_name_bge="BAAI/bge-m3",
        model_weight="visualized_m3.pth",
        image_tokens_num=256,
    ),
    name="BAAI/bge-visualized-m3",
    languages=["eng_Latn"],
    revision="98db10b10d22620010d06f11733346e1c98c34aa",
    release_date="2024-06-06",
    modalities=["image", "text"],
    n_parameters=872_909_505,
    memory_usage_mb=4263,
    max_tokens=8192,
    embed_dim=1024,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/JUNJIE99/VISTA_S2",
    framework=["PyTorch"],
    reference="https://huggingface.co/BAAI/bge-visualized",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=vista_training_datasets,
)
