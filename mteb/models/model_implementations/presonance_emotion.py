"""Image-only Presonance EmoSet encoder for MTEB. Fixed public checkpoints; no task-specific neural fitting."""

import hashlib
import json
from pathlib import Path

import numpy as np
from mteb.models import ModelMeta
from mteb.models.model_meta import ScoringFunction


def unit(features):
    x = np.asarray(features, dtype=np.float32)
    den = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and np.isfinite(den).all() and (den > 0).all()
    return x / den


def combine(blocks):
    assert len(blocks) == 4
    return np.concatenate([unit(b) for b in blocks], axis=1) * np.float32(0.5)


def feature_batch(model, batch):
    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = model.forward_head(model.forward_features(batch), pre_logits=True).float()
        b = model.forward_head(
            model.forward_features(torch.flip(batch, dims=[3])), pre_logits=True
        ).float()
    return ((a + b) / 2).cpu().numpy()


CHECKPOINT_REPO = "saa938/presonance-emoset-ensemble"
CHECKPOINT_REVISION = "1c9fd34ff2b79daed84239237fa941fed17652d1"


class PresonanceEmotionEncoder:
    def __init__(
        self,
        model_name=CHECKPOINT_REPO,
        revision=CHECKPOINT_REVISION,
        *,
        device="cuda",
        checkpoint_dir=None,
        **kwargs,
    ):
        if checkpoint_dir is None:
            from huggingface_hub import snapshot_download

            checkpoint_dir = snapshot_download(
                CHECKPOINT_REPO, revision=CHECKPOINT_REVISION
            )
        self.root = Path(checkpoint_dir)
        self.device = device
        self.config = json.loads((self.root / "config.json").read_text())
        assert [m["name"] for m in self.config["models"]] == [
            "clipL336_e6",
            "eva02L336_e6",
            "pecoreL336_e6",
            "siglip2giant384_e6",
        ]

    def predict_features(self, images, batch_size=64):
        import timm
        import torch
        from safetensors.torch import load_file
        from timm.data import create_transform

        assert self.device.startswith("cuda") and batch_size == 64, (
            "Validated condition: CUDA BF16, batch size64"
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        blocks = []
        for spec in self.config["models"]:
            path = self.root / spec["file"]
            with path.open("rb") as f:
                assert hashlib.file_digest(f, "sha256").hexdigest() == spec["sha256"]
            model = timm.create_model(
                spec["architecture"], pretrained=False, num_classes=8
            )
            model.load_state_dict(
                {k: v.float() for k, v in load_file(str(path)).items()}, strict=True
            )
            model.eval().requires_grad_(False).to(self.device)
            tf = create_transform(is_training=False, **spec["preprocessing"])
            rows = []
            for at in range(0, len(images), 64):
                batch = torch.stack(
                    [tf(im.convert("RGB")) for im in images[at : at + 64]]
                ).to(self.device)
                rows.append(feature_batch(model, batch))
            blocks.append(np.concatenate(rows))
            del model
            torch.cuda.empty_cache()
        return combine(blocks)

    def encode(self, inputs, **kwargs):
        images = []
        for batch in inputs:
            images.extend(batch["image"])
        return self.predict_features(images, batch_size=64)

    def similarity(self, a, b):
        return unit(a) @ unit(b).T

    def similarity_pairwise(self, a, b):
        return np.sum(unit(a) * unit(b), axis=1)

    @property
    def mteb_model_meta(self):
        return presonance_emoset_image


def _load_encoder(model_name, revision, **kwargs):
    return PresonanceEmotionEncoder(model_name, revision, **kwargs)


presonance_emoset_image = ModelMeta(
    loader=_load_encoder,
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
