from __future__ import annotations

from functools import partial
from typing import Any, List

import torch
from PIL import Image
from tqdm import tqdm

from mmE5.src.model import MMEBModel
from mmE5.src.arguments import ModelArguments
from mmE5.src.utils import load_processor

from mteb.model_meta import ModelMeta


class MMEBModelWrapper:
    """
    A wrapper for the intfloat/mmE5-mllama-11b-instruct model.
    
    This model can encode both text and images.
    """

    def __init__(
        self,
        model_name: str = "intfloat/mmE5-mllama-11b-instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pooling: str = "last",
        normalize: bool = True,
        model_backbone: str = "mllama",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device

        model_args = ModelArguments(
            model_name=model_name,
            pooling=pooling,
            normalize=normalize,
            model_backbone=model_backbone,
        )
        self.processor = load_processor(model_args)
        self.model = MMEBModel.load(model_args)
        self.model.eval()
        self.model = self.model.to(self.device, dtype=torch.bfloat16)

    def preprocess(
        self, texts: List[str] = None, images: List[Image.Image] = None, **kwargs: Any
    ):
        """
        Preprocess texts and images using the model's processor.
        """
        return self.processor(text=texts, images=images, return_tensors="pt", padding=True)

    def get_text_embeddings(
        self,
        texts: List[str],
        mode: str = "tgt",
        batch_size: int = 32,
        **kwargs: Any,
    ):
        """
        Compute text embeddings.
        
        The model expects text inputs on the target branch (using key "tgt").
        If needed, you can switch to the query branch by setting mode="qry".
        """
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Text Embeddings"):
                batch_texts = texts[i : i + batch_size]
                inputs = self.processor(
                    text=batch_texts, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**{mode: inputs})
                embeddings = outputs[f"{mode}_reps"]
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def get_image_embeddings(
        self,
        images: List[Image.Image],
        mode: str = "tgt",
        prompt: str = "<|image|><|begin_of_text|> Represent the given image.",
        batch_size: int = 32,
        **kwargs: Any,
    ):
        """
        Compute image embeddings.
        
        The processor requires a text prompt alongside images.
        """
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="Image Embeddings"):
                batch_images = images[i : i + batch_size]
                inputs = self.processor(
                    text=prompt, images=batch_images, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**{mode: inputs})
                embeddings = outputs[f"{mode}_reps"]
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def get_fused_embeddings(
        self,
        texts: List[str] = None,
        images: List[Image.Image] = None,
        fusion_mode: str = "sum",
        batch_size: int = 32,
        **kwargs: Any,
    ):
        """
        Compute fused embeddings by combining text and image representations.
        
        Currently, only the elementwise sum fusion is implemented.
        """
        text_embeddings = None
        image_embeddings = None
        if texts is not None:
            text_embeddings = self.get_text_embeddings(texts, batch_size=batch_size, **kwargs)
        if images is not None:
            image_embeddings = self.get_image_embeddings(images, batch_size=batch_size, **kwargs)
        if text_embeddings is not None and image_embeddings is not None:
            if text_embeddings.shape[0] != image_embeddings.shape[0]:
                raise ValueError("The number of texts and images must be equal for fusion.")
            if fusion_mode == "sum":
                return text_embeddings + image_embeddings
            else:
                raise ValueError(f"Fusion mode '{fusion_mode}' is not implemented.")
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        else:
            raise ValueError("At least one of texts or images must be provided.")

    def compute_similarity(
        self, query_embeddings: torch.Tensor, candidate_embeddings: torch.Tensor
    ):
        """
        Compute cosine similarity between query and candidate embeddings.
        """
        similarity = self.model.compute_similarity(query_embeddings, candidate_embeddings)
        return similarity


mme5_model_meta = ModelMeta(
    loader=partial(
        MMEBModelWrapper,
        model_name="intfloat/mmE5-mllama-11b-instruct",
    ),
    name="intfloat/mmE5-mllama-11b-instruct",
    languages=["eng_Latn"],
    revision=None,
    release_date="2025-02-16",
    modalities=["image", "text"],
    n_parameters=10_600_000_000,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=None,
    license=None,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/intfloat/mmE5-mllama-11b-instruct",
    similarity_fn_name="compute_similarity",
    use_instructions=True,
    training_datasets=None,
)
