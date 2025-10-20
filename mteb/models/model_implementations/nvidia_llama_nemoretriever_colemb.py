from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, BatchedInput, PromptType

LLAMA_NEMORETRIEVER_CITATION = """@misc{xu2025llamanemoretrievercolembedtopperforming,
      title={Llama Nemoretriever Colembed: Top-Performing Text-Image Retrieval Model},
      author={Mengyao Xu and Gabriel Moreira and Ronay Ak and Radek Osmulski and Yauhen Babakhin and Zhiding Yu and Benedikt Schifferer and Even Oldridge},
      year={2025},
      eprint={2507.05513},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05513}
}"""


class LlamaNemoretrieverColembed(AbsEncoder):
    def __init__(
        self,
        model_name_or_path: str,
        revision: str,
        trust_remote_code: bool,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        **kwargs,
    ):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
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
            raise NotImplementedError(
                "Fused embeddings are not supported yet. Please use get_text_embeddings or get_image_embeddings."
            )
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        raise ValueError


TRAINING_DATA = {
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "DocVQA",
    "InfoVQA",
    "TATDQA",
    "arXivQA",
    "hotpotqa",
    "miracl",
    "NQ",
    "stackexchange",
    "SQuAD",
    "WebInstructSub",
    "docmatix-ir",
    "vdr-multilingual-train",
    "colpali_train_set",  # as it contains PDFs
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
    "wiki-ss-nq",
}

llama_nemoretriever_colembed_1b_v1 = ModelMeta(
    loader=LlamaNemoretrieverColembed,
    loader_kwargs=dict(
        trust_remote_code=True,
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
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=LLAMA_NEMORETRIEVER_CITATION,
)

llama_nemoretriever_colembed_3b_v1 = ModelMeta(
    loader=LlamaNemoretrieverColembed,
    loader_kwargs=dict(
        trust_remote_code=True,
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
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=LLAMA_NEMORETRIEVER_CITATION,
)
