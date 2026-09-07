"""MLCD image encoders (DeepGlint-AI) - image-only CLIPVisionModel-compatible checkpoints.

Feature endpoint: the public HF checkpoints contain only the vision tower (verified by
weight inspection - no visual_projection, no unicom embedding head), so embeddings are
the pooled CLS token after post_layernorm (768-d base, 1024-d large).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput


class MLCDVisionModel(AbsEncoder):
    """Image-only encoder: pooled CLS vision features from a CLIPVisionModel checkpoint."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from transformers import CLIPImageProcessor, CLIPVisionModel

        self.model_name = model_name
        self.device = device
        self.model = CLIPVisionModel.from_pretrained(
            model_name, revision=revision
        ).to(self.device)
        self.processor = CLIPImageProcessor.from_pretrained(model_name, revision=revision)
        self.model.eval()

    @torch.no_grad()
    def get_image_embeddings(
        self,
        images: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        all_image_embeddings = []
        for batch in tqdm(images, disable=not show_progress_bar, desc="Image Encoding"):
            if "image" not in batch:
                raise ValueError(
                    "MLCD is image-only: batch has no 'image' field; "
                    f"got keys {list(batch.keys())}"
                )
            imgs = batch["image"]
            if any(im is None for im in imgs):
                raise ValueError("MLCD is image-only: a required image is missing (None).")
            inputs = self.processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            pooled = outputs.pooler_output  # CLS after post_layernorm
            all_image_embeddings.append(pooled.cpu())

        return torch.cat(all_image_embeddings, dim=0)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type=None,
        **kwargs: Any,
    ) -> Array:
        if "image" in inputs.dataset.features:
            return self.get_image_embeddings(inputs, **kwargs)
        raise ValueError(
            "MLCD is image-only: no 'image' feature in input batch; refusing to "
            "substitute text or other modalities."
        )


MLCD_CITATION = """
@article{yin2024mlcd,
  title={Multi-Label Cluster Discrimination for Visual Representation Learning},
  author={Yin, Xie and Wang, Yumeng and Cao, Jia and Wang, Qi and Wang, Di and others},
  journal={arXiv preprint arXiv:2407.17331},
  year={2024}
}"""

_common = dict(
    loader=MLCDVisionModel,
    model_type=["dense"],
    languages=["eng-Latn"],
    modalities=["image"],
    open_weights=True,
    framework=["PyTorch", "Transformers"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    citation=MLCD_CITATION,
    max_tokens=77,
)

mlcd_vit_base_patch32_224 = ModelMeta(
    **_common,
    name="DeepGlint-AI/mlcd-vit-base-patch32-224",
    revision="0862ef00ecb5825c1c79ab5944ceea56a5605113",
    release_date="2024-11-04",
    n_parameters=87456000,
    embed_dim=768,
    memory_usage_mb=335,
    license="apache-2.0",
    reference="https://huggingface.co/DeepGlint-AI/mlcd-vit-base-patch32-224",
    public_training_code="https://github.com/deepglint/unicom",
    public_training_data="https://huggingface.co/datasets/kakaobrain/coyo-700m",
    training_datasets={"COYO700M"},
)

mlcd_vit_large_patch14_336 = ModelMeta(
    **_common,
    name="DeepGlint-AI/mlcd-vit-large-patch14-336",
    revision="bd75627f99f12c1d2e0a40ed8c9d1f129b54405e",
    release_date="2024-11-13",
    n_parameters=303507456,
    embed_dim=1024,
    memory_usage_mb=1157,
    license="apache-2.0",
    reference="https://huggingface.co/DeepGlint-AI/mlcd-vit-large-patch14-336",
    public_training_code="https://github.com/deepglint/unicom",
    public_training_data="https://huggingface.co/datasets/laion/laion400m",
    training_datasets={"LAION400M", "COYO700M"},
)
