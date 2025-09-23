from __future__ import annotations

from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModel

from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class llama_nemoretriever_colembed(Wrapper):
    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        revision=None,
        **kwargs,
    ):
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            revision=revision,
        ).eval()

    def get_text_embeddings(self, texts, batch_size: int = 32, **kwargs):
        batch_size = 1
        return self.model.forward_queries(texts, batch_size=batch_size)

    def get_image_embeddings(
        self,
        images,
        batch_size: int = 32,
        **kwargs,
    ):
        import torchvision.transforms.functional as F

        all_images = []
        if isinstance(images, DataLoader):
            iterator = images
        else:
            iterator = DataLoader(images, batch_size=batch_size)

        for batch in iterator:
            for b in batch:
                pil_img = (
                    F.to_pil_image(b.to("cpu")) if not isinstance(b, Image.Image) else b
                )
                all_images.append(pil_img)

        batch_size = 1
        return self.model.forward_passages(all_images, batch_size=batch_size)

    def calculate_probs(self, text_embeddings, image_embeddings):
        scores = self.similarity(text_embeddings, image_embeddings)
        return (scores * 100).softmax(dim=-1)

    def similarity(self, a, b):
        return self.model.get_scores(a, b)

    def get_fused_embeddings(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            "Fused embeddings are not supported yet. Please use get_text_embeddings or get_image_embeddings."
        )

    def encode(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            "Encode is not implemented. Please use .mdl.forward_queries or mdl.forward_passages."
        )


TRAINING_DATA = {
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "DocVQA": ["train"],
    "InfoVQA": ["train"],
    "TATDQA": ["train"],
    "arXivQA": ["train"],
    "hotpotqa": ["train"],
    "miracl": ["train"],
    "NQ": ["train"],
    "stackexchange": ["train"],
    "SQuAD": ["train"],
    "WebInstructSub": ["train"],
    "docmatix-ir": ["train"],
    "vdr-multilingual-train": ["train"],
    "colpali_train_set": ["train"],  # as it contains PDFs
    "VisRAG-Ret-Train-Synthetic-data": ["train"],
    "VisRAG-Ret-Train-In-domain-data": ["train"],
    "wiki-ss-nq": ["train"],
}

llama_nemoretriever_colembed_1b_v1 = ModelMeta(
    loader=partial(
        llama_nemoretriever_colembed,
        model_name_or_path="nvidia/llama-nemoretriever-colembed-1b-v1",
        trust_remote_code=True,
        revision="1f0fdea7f5b19532a750be109b19072d719b8177",
    ),
    name="nvidia/llama-nemoretriever-colembed-1b-v1",
    languages=["eng-Latn"],
    revision="1f0fdea7f5b19532a750be109b19072d719b8177",
    release_date="2025-06-27",
    modalities=["image", "text"],
    n_parameters=2_418_000_000,
    memory_usage_mb=9224,
    max_tokens=8192,
    embed_dim=2048,
    license="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1/blob/main/LICENSE",
    open_weights=True,
    public_training_code="Proprietary Code",
    public_training_data="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1#training-dataset",
    framework=["PyTorch"],
    reference="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=TRAINING_DATA,
)

llama_nemoretriever_colembed_3b_v1 = ModelMeta(
    loader=partial(
        llama_nemoretriever_colembed,
        model_name_or_path="nvidia/llama-nemoretriever-colembed-3b-v1",
        trust_remote_code=True,
        revision="50c36f4d5271c6851aa08bd26d69f6e7ca8b870c",
    ),
    name="nvidia/llama-nemoretriever-colembed-3b-v1",
    languages=["eng-Latn"],
    revision="50c36f4d5271c6851aa08bd26d69f6e7ca8b870c",
    release_date="2025-06-27",
    modalities=["image", "text"],
    n_parameters=4_407_000_000,
    memory_usage_mb=16811,
    max_tokens=8192,
    embed_dim=3072,
    license="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1/blob/main/LICENSE",
    open_weights=True,
    public_training_code="Proprietary Code",
    public_training_data="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1#training-dataset",
    framework=["PyTorch"],
    reference="https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1",
    similarity_fn_name="max_sim",
    use_instructions=True,
    training_datasets=TRAINING_DATA,
)
