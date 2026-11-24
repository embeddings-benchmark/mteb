"""Transformers wrapper for the separately licensed trained metric image encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import override

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    import numpy as np
    from torch.utils.data import DataLoader

    from mteb.types import Array, BatchedInput


def _numpy_matrix(values: Array) -> np.ndarray:
    import numpy as np
    import torch

    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    return np.atleast_2d(np.asarray(values, dtype=np.float32))


class PresonanceMetricVisionEncoder(AbsEncoder):
    """Load a trained metric image encoder through Transformers."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            code_revision=revision,
            trust_remote_code=True,
            device=device or "cuda",
            cache_dir=cache_dir,
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode images using the qualified image representation."""
        return self.model.encode(inputs, batch_size=batch_size)

    @override
    def similarity(self, embeddings1: Array, embeddings2: Array) -> np.ndarray:
        # Preserve the evaluated NumPy dot product instead of the inherited
        # accelerator path, which can use reduced-precision matrix products.
        return _numpy_matrix(embeddings1) @ _numpy_matrix(embeddings2).T

    @override
    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> np.ndarray:
        return (_numpy_matrix(embeddings1) * _numpy_matrix(embeddings2)).sum(axis=1)


presonance_generalist_vision_metric_v1 = ModelMeta(
    loader=PresonanceMetricVisionEncoder,
    extra_requirements_groups=["presonance-metric-vision"],
    name="saa938/presonance-generalist-vision-metric-v1",
    revision="a90033012c4fb290b45baa891b167a08f697583c",
    release_date="2026-11-24",
    languages=["eng-Latn"],
    n_parameters=5_192_213_632,
    memory_usage_mb=19807,
    max_tokens=None,
    embed_dim=4480,
    license="https://huggingface.co/saa938/presonance-generalist-vision-metric-v1/blob/a90033012c4fb290b45baa891b167a08f697583c/LICENSE",
    open_weights=True,
    public_training_code="https://huggingface.co/saa938/presonance-generalist-vision-metric-v1/blob/a90033012c4fb290b45baa891b167a08f697583c/training_recipe.py",
    public_training_data=None,
    framework=["Transformers", "PyTorch"],
    reference="https://huggingface.co/saa938/presonance-generalist-vision-metric-v1",
    similarity_fn_name=ScoringFunction.DOT_PRODUCT,
    use_instructions=False,
    training_datasets={
        "FER2013",
        "Caltech101",
        "OxfordPets",
        "OxfordFlowersClassification",
        "DTD",
        "CIFAR10",
        "STL10",
        "EuroSAT",
        "GTSRB",
    },
    modalities=["image"],
    model_type=["dense"],
)


presonance_generalist_vision_geometry_v1 = ModelMeta(
    loader=PresonanceMetricVisionEncoder,
    extra_requirements_groups=["presonance-metric-vision"],
    name="saa938/presonance-generalist-vision-geometry-v1",
    revision="4a1d52b2642ae25285986f29f97504a5a389799a",
    release_date="2026-11-24",
    languages=["eng-Latn"],
    n_parameters=5_192_213_632,
    memory_usage_mb=19807,
    max_tokens=None,
    embed_dim=4480,
    license="https://huggingface.co/saa938/presonance-generalist-vision-geometry-v1/blob/4a1d52b2642ae25285986f29f97504a5a389799a/LICENSE",
    open_weights=True,
    public_training_code="https://huggingface.co/saa938/presonance-generalist-vision-geometry-v1/blob/4a1d52b2642ae25285986f29f97504a5a389799a/training_recipe.py",
    public_training_data=None,
    framework=["Transformers", "PyTorch"],
    reference="https://huggingface.co/saa938/presonance-generalist-vision-geometry-v1",
    similarity_fn_name=ScoringFunction.DOT_PRODUCT,
    use_instructions=False,
    training_datasets={
        "FER2013",
        "Caltech101",
        "OxfordPets",
        "OxfordFlowersClassification",
        "DTD",
        "CIFAR10",
        "STL10",
        "EuroSAT",
        "GTSRB",
    },
    modalities=["image"],
    model_type=["dense"],
)
