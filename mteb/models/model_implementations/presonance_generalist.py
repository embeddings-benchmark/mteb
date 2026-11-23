"""Checksum-pinned benchmark wrapper for the separately licensed reference encoder."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    import numpy as np
    from torch.utils.data import DataLoader

    from mteb.types import Array, BatchedInput

MODEL_REPO = "saa938/presonance-generalist-vision-v1"
MODEL_REVISION = "e2c798553cc653a2c38e8daa22ac93669cbaf9af"
ENCODER_SHA256 = "94baaaa35a6a3c8e29856875c0f06930f5a5e8dc1696cdeccd3e29788e4c89e2"
CONFIG_SHA256 = "f57cc22417736f6e1704ccbcbf602eb120beef31c69b1600bc265c65b10bbe96"


def _numpy_matrix(values: Array) -> np.ndarray:
    import numpy as np
    import torch

    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    return np.atleast_2d(np.asarray(values, dtype=np.float32))


class PresonanceGeneralistVisionEncoder(AbsEncoder):
    """Load the fixed, source-verified image encoder."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                model_name,
                revision=revision,
                allow_patterns=["config.json", "encoder.py"],
                cache_dir=cache_dir,
            )
        )
        for name, expected in [
            ("encoder.py", ENCODER_SHA256),
            ("config.json", CONFIG_SHA256),
        ]:
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError("The reference encoder or configuration differs")
        # The reference implementation retains its own research license. Keep
        # this integration separate and load only its exact reviewed bytes.
        spec = importlib.util.spec_from_file_location(
            "presonance_generalist_reference", root / "encoder.py"
        )
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load the pinned reference implementation")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.PresonanceGeneralistEncoder(
            root / "config.json", device=device or "cuda", cache_dir=cache_dir
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
    framework=["PyTorch"],
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
