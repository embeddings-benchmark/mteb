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


class NemotronColEmbedVLV2(AbsEncoder):
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

        # Check if required packages are installed
        requires_package(
            self,
            "torchvision",
            model_name_or_path,
            "pip install 'mteb[nemotron-colembed-vl-v2]'",
        )
        requires_package(
            self,
            "flash_attn",
            model_name_or_path,
            "pip install 'mteb[nemotron-colembed-vl-v2]'",
        )
        import flash_attn  # noqa: F401
        import torchvision  # noqa: F401

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

nemotron_colembed_vl_4b_v2 = ModelMeta(
    loader=NemotronColEmbedVLV2,
    loader_kwargs=dict(
        trust_remote_code=True,
        transformers_version_constraint=">=5.0.0",
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
    loader=NemotronColEmbedVLV2,
    loader_kwargs=dict(
        trust_remote_code=True,
        transformers_version_constraint=">=5.0.0",
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
