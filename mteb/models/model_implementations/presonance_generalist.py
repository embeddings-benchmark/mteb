"""Transformers wrapper for the separately licensed fixed image encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import override

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    import numpy as np
    from torch.utils.data import DataLoader

    from mteb.types import Array, BatchedInput

MODEL_REPO = "saa938/presonance-generalist-vision-v1"
MODEL_REVISION = "73399519ea350e346ef6ff7ee4ba831c0d8bd5fc"


def _numpy_matrix(values: Array) -> np.ndarray:
    import numpy as np
    import torch

    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    return np.atleast_2d(np.asarray(values, dtype=np.float32))


class PresonanceGeneralistVisionEncoder(AbsEncoder):
    """Load the fixed image encoder through Transformers."""

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
        """Encode images using the qualified fixed representation."""
        return self.model.encode(inputs, batch_size=batch_size)

    @override
    def similarity(self, embeddings1: Array, embeddings2: Array) -> np.ndarray:
        # Preserve the evaluated NumPy dot product instead of the inherited
        # accelerator path, which can use reduced-precision matrix products.
        return _numpy_matrix(embeddings1) @ _numpy_matrix(embeddings2).T

    @override
    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> np.ndarray:
        return (_numpy_matrix(embeddings1) * _numpy_matrix(embeddings2)).sum(axis=1)


presonance_generalist_vision_v1 = ModelMeta(
    loader=PresonanceGeneralistVisionEncoder,
    extra_requirements_groups=["presonance-generalist-vision"],
    name="saa938/presonance-generalist-vision-v1",
    revision=MODEL_REVISION,
    release_date="2026-11-23",
    languages=["eng-Latn"],
    n_parameters=5_191_066_368,
    memory_usage_mb=19802,
    max_tokens=None,
    embed_dim=4352,
    # Custom noncommercial code terms; upstream checkpoint terms also apply.
    license="https://huggingface.co/" + MODEL_REPO + "/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Transformers", "PyTorch"],
    reference="https://huggingface.co/" + MODEL_REPO,
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
