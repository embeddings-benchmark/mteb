from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from packaging.specifiers import SpecifierSet
from torch.utils.data import DataLoader
from transformers import __version__ as transformers_version

from mteb._requires_package import requires_package
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

# Transformers version constraints per extra.
# Keep in sync with pyproject.toml [project.optional-dependencies]
#
# Note: The extra name reflects the transformers version requirement, not the model version.
# For example, llama-nemotron-colembed-vl-3b-v2 uses "nemotron-colembed-vl" because it
# currently requires transformers==4.49.0, even though it's a "v2" model by name.
_TRANSFORMERS_CONSTRAINTS: dict[str, str] = {
    "nemotron-colembed-vl": "==4.49.0",  # llama-nemoretriever-colembed-*-v1, llama-nemotron-colembed-vl-3b-v2
    "nemotron-colembed-vl-v2": ">=5.0.0",  # nemotron-colembed-vl-4b-v2, nemotron-colembed-vl-8b-v2
}


class NemotronColEmbedVL(AbsEncoder):
    """Encoder for the NemotronColEmbedVL family of models."""

    def __init__(
        self,
        model_name_or_path: str,
        revision: str,
        trust_remote_code: bool,
        extra_name: str = "nemotron-colembed-vl",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        **kwargs,
    ):
        install_hint = f"pip install 'mteb[{extra_name}]'"

        # Check transformers version
        constraint = _TRANSFORMERS_CONSTRAINTS.get(extra_name)
        if constraint is None:
            raise ValueError(
                f"Unknown extra_name '{extra_name}'. "
                f"Must be one of: {list(_TRANSFORMERS_CONSTRAINTS.keys())}"
            )
        if transformers_version not in SpecifierSet(constraint):
            raise RuntimeError(
                f"Model `{model_name_or_path}` requires transformers{constraint}, "
                f"but {transformers_version} is installed. "
                f"Run: {install_hint}"
            )

        # Check required packages
        for package in ("torchvision", "accelerate", "flash_attn"):
            requires_package(self, package, model_name_or_path, install_hint)

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
    loader=NemotronColEmbedVL,
    loader_kwargs=dict(
        extra_name="nemotron-colembed-vl",
        trust_remote_code=True,
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
    loader=NemotronColEmbedVL,
    loader_kwargs=dict(
        extra_name="nemotron-colembed-vl",
        trust_remote_code=True,
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
    loader=NemotronColEmbedVL,
    loader_kwargs=dict(
        extra_name="nemotron-colembed-vl",
        trust_remote_code=True,
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
    loader=NemotronColEmbedVL,
    loader_kwargs=dict(
        extra_name="nemotron-colembed-vl-v2",
        trust_remote_code=True,
    ),
    name="nvidia/nemotron-colembed-vl-4b-v2",
    revision="df61c25b416c1ef27cd0d7039596f752c262c5ae",
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
    citation=None,
)


nemotron_colembed_vl_8b_v2 = ModelMeta(
    loader=NemotronColEmbedVL,
    loader_kwargs=dict(
        extra_name="nemotron-colembed-vl-v2",
        trust_remote_code=True,
    ),
    name="nvidia/nemotron-colembed-vl-8b-v2",
    revision="b6db318a581be528d8a24c25f0857d7d833d5263",
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
    citation=None,
)
