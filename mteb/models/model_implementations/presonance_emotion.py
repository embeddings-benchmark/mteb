"""Image-only Presonance EmoSet encoder using fixed public checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from mteb.types import BatchedInput
    from numpy.typing import ArrayLike
    from PIL import Image
    from torch.utils.data import DataLoader


def unit(features: ArrayLike) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    den = np.linalg.norm(x, axis=1, keepdims=True)
    if not (np.isfinite(x).all() and np.isfinite(den).all() and (den > 0).all()):
        raise ValueError("Image features must contain finite, nonzero vectors.")
    return x / den


def combine(blocks: Sequence[np.ndarray]) -> np.ndarray:
    if len(blocks) != 4:
        raise ValueError("The Presonance encoder requires four feature blocks.")
    return np.concatenate([unit(b) for b in blocks], axis=1) * np.float32(0.5)


def feature_batch(model: torch.nn.Module, batch: torch.Tensor) -> np.ndarray:
    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = model.forward_head(model.forward_features(batch), pre_logits=True).float()
        b = model.forward_head(
            model.forward_features(torch.flip(batch, dims=[3])), pre_logits=True
        ).float()
    return ((a + b) / 2).cpu().numpy()


CHECKPOINT_REPO = "saa938/presonance-emoset-ensemble"
CHECKPOINT_REVISION = "1c9fd34ff2b79daed84239237fa941fed17652d1"


class PresonanceEmotionEncoder(AbsEncoder):
    def __init__(
        self,
        model_name: str = CHECKPOINT_REPO,
        revision: str = CHECKPOINT_REVISION,
        *,
        device: str | None = "cuda",
        checkpoint_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        if checkpoint_dir is None:
            from huggingface_hub import snapshot_download

            checkpoint_dir = snapshot_download(
                CHECKPOINT_REPO, revision=CHECKPOINT_REVISION
            )
        self.root = Path(checkpoint_dir)
        self.device = device or "cuda"
        self.config = json.loads((self.root / "config.json").read_text())
        if [m["name"] for m in self.config["models"]] != [
            "clipL336_e6",
            "eva02L336_e6",
            "pecoreL336_e6",
            "siglip2giant384_e6",
        ]:
            raise ValueError("Checkpoint configuration has an unexpected model order.")

    def predict_features(
        self, images: Sequence[Image.Image], batch_size: int = 64
    ) -> np.ndarray:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")
        if not self.device.startswith("cuda"):
            raise ValueError("This encoder requires CUDA for BF16 image inference.")

        import timm
        import torch
        from safetensors.torch import load_file
        from timm.data import create_transform

        # The published run used 64. Other positive batch sizes are configurable;
        # no bitwise parity across CUDA batch sizes is implied.
        torch.backends.cuda.matmul.allow_tf32 = True
        blocks = []
        for spec in self.config["models"]:
            path = self.root / spec["file"]
            with path.open("rb") as f:
                actual_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
            if actual_sha256 != spec["sha256"]:
                raise ValueError(f"Checkpoint SHA256 mismatch: {spec['file']}")
            model = timm.create_model(
                spec["architecture"], pretrained=False, num_classes=8
            )
            model.load_state_dict(
                {k: v.float() for k, v in load_file(str(path)).items()}, strict=True
            )
            model.eval().requires_grad_(False).to(self.device)
            tf = create_transform(is_training=False, **spec["preprocessing"])
            rows = []
            for at in range(0, len(images), batch_size):
                batch = torch.stack(
                    [tf(im.convert("RGB")) for im in images[at : at + batch_size]]
                ).to(self.device)
                rows.append(feature_batch(model, batch))
            blocks.append(np.concatenate(rows))
            del model
            torch.cuda.empty_cache()
        return combine(blocks)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> np.ndarray:
        images = []
        for batch in inputs:
            images.extend(batch["image"])
        return self.predict_features(images, batch_size=batch_size)


def _load_encoder(
    model_name: str, revision: str, **kwargs: Any
) -> PresonanceEmotionEncoder:
    return PresonanceEmotionEncoder(model_name, revision, **kwargs)


presonance_emoset_image = ModelMeta(
    loader=_load_encoder,
    extra_requirements_groups=["timm"],
    name=CHECKPOINT_REPO,
    revision=CHECKPOINT_REVISION,
    release_date="2026-11-14",
    languages=["eng-Latn"],
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=4608,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data="https://vcc.tech/EmoSet",
    framework=["PyTorch"],
    reference="https://huggingface.co/" + CHECKPOINT_REPO,
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets={"EmoSet"},
    modalities=["image"],
    model_type=["dense"],
)
