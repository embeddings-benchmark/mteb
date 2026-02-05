from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from packaging.specifiers import SpecifierSet
from torch.utils.data import DataLoader
from transformers import __version__ as transformers_version

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
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
        transformers_version_constraint: str | None = None,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        **kwargs,
    ):
        if transformers_version_constraint is not None:
            spec = SpecifierSet(transformers_version_constraint)
            if transformers_version not in spec:
                raise RuntimeError(
                    f"Model `{model_name_or_path}` requires transformers{transformers_version_constraint}, "
                    f"but {transformers_version} is installed. "
                    f"Run: pip install 'transformers{transformers_version_constraint}'"
                )

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
        from PIL import Image

        all_images = []
        if isinstance(images, DataLoader):
            iterator = images
        else:
            iterator = DataLoader(images, batch_size=batch_size)

        for batch in iterator:
            for image in batch["image"]:
                pil_img = (
                    image
                    if isinstance(image, Image.Image)
                    else F.to_pil_image(image.to("cpu"))
                )
                all_images.append(pil_img)

        batch_size = 1
        return self.model.forward_images(all_images, batch_size=batch_size)

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
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "HotpotQA",
    "MIRACLRetrieval",
    "NQ",
    "StackExchangeClustering",
    "SQuAD",
    "WebInstructSub",
    "docmatix-ir",
    "VDRMultilingualRetrieval",
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
    "wiki-ss-nq",
}


TRAINING_DATA_v2 = {
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "docmatix-ir",
    "VDRMultilingualRetrieval",
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
    "wiki-ss-nq",
}

llama_nemoretriever_colembed_1b_v1 = ModelMeta(
    loader=LlamaNemoretrieverColembed,
    loader_kwargs=dict(
        trust_remote_code=True,
        transformers_version_constraint="==4.49.0",
    ),
    name="nvidia/llama-nemoretriever-colembed-1b-v1",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="6eade800103413033f260bb55b49fe039fd28a6e",
    release_date="2025-06-27",
    modalities=["image", "text"],
    n_parameters=2_418_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=4610,
    max_tokens=8192,
    embed_dim=2048,
    license="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1#training-dataset",
    framework=["PyTorch", "Transformers", "safetensors"],
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
        transformers_version_constraint="==4.49.0",
    ),
    name="nvidia/llama-nemoretriever-colembed-3b-v1",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="4194bdd2cd2871f220ddba6273ce173ef1217a1e",
    release_date="2025-06-27",
    modalities=["image", "text"],
    n_parameters=4_407_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=8403,
    max_tokens=8192,
    embed_dim=3072,
    license="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/llama-nemoretriever-colembed-1b-v1#training-dataset",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=LLAMA_NEMORETRIEVER_CITATION,
)

llama_nemotron_colembed_vl_3b_v2 = ModelMeta(
    loader=LlamaNemoretrieverColembed,
    loader_kwargs=dict(
        trust_remote_code=True,
        transformers_version_constraint="==4.49.0",
    ),
    name="nvidia/llama-nemotron-colembed-vl-3b-v2",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="75f03c712cb3a252e062295f9a0966e5d95d6156",
    release_date="2026-01-21",
    modalities=["image", "text"],
    n_parameters=4_407_000_000,
    memory_usage_mb=8403,
    max_tokens=8192,
    embed_dim=3072,
    license="https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2#training-dataset",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=TRAINING_DATA,
    citation=LLAMA_NEMORETRIEVER_CITATION,
)

nemotron_colembed_vl_4b_v2 = ModelMeta(
    loader=LlamaNemoretrieverColembed,
    loader_kwargs=dict(
        trust_remote_code=True,
        transformers_version_constraint="==5.0.0rc0",
    ),
    name="nvidia/nemotron-colembed-vl-4b-v2",
    revision="823b1625c15fe3da73fa094205e538a7a2301a2a",
    languages=["eng-Latn"],
    release_date="2026-01-07",
    modalities=["image", "text"],
    n_parameters=4_800_000_000,
    memory_usage_mb=9206,
    max_tokens=262144,
    embed_dim=2560,
    license="https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2#training-dataset",
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=TRAINING_DATA_v2,
    citation=LLAMA_NEMORETRIEVER_CITATION,
)


nemotron_colembed_vl_8b_v2 = ModelMeta(
    loader=LlamaNemoretrieverColembed,
    loader_kwargs=dict(
        trust_remote_code=True,
        transformers_version_constraint="==5.0.0rc0",
    ),
    name="nvidia/nemotron-colembed-vl-8b-v2",
    revision="6cbe43579dda6237768fc373768ad372cc5cdfec",
    languages=["eng-Latn"],
    release_date="2026-01-07",
    modalities=["image", "text"],
    n_parameters=8_700_000_000,
    memory_usage_mb=16722,
    max_tokens=262144,
    embed_dim=4096,
    license="https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2#training-dataset",
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=TRAINING_DATA_v2,
    citation=LLAMA_NEMORETRIEVER_CITATION,
)
