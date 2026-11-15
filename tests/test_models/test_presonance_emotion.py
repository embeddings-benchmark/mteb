import hashlib
import json
import sys
import types
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta

from mteb.models.model_implementations import presonance_emotion as encoder_module


@pytest.fixture
def checkpoints(tmp_path):
    names = ["clipL336_e6", "eva02L336_e6", "pecoreL336_e6", "siglip2giant384_e6"]
    specs = []
    for index, name in enumerate(names):
        raw = f"synthetic checkpoint {index}".encode()
        path = tmp_path / f"model_{index}.safetensors"
        path.write_bytes(raw)
        specs.append(
            {
                "name": name,
                "file": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "architecture": f"synthetic_{index}",
                "preprocessing": {},
            }
        )
    (tmp_path / "config.json").write_text(json.dumps({"models": specs}))
    return tmp_path


def test_official_metadata_attachment_and_inherited_cosine(checkpoints):
    meta = encoder_module.presonance_emoset_image
    # The test concerns the loader contract, without installing neural extras.
    with patch.object(ModelMeta, "_check_requirements"):
        model = meta.load_model(checkpoint_dir=checkpoints, device="cuda:0")
    assert isinstance(model, AbsEncoder)
    assert model.device == "cuda:0"
    assert model.mteb_model_meta.name == meta.name
    assert model.mteb_model_meta is not meta
    assert "mteb_model_meta" not in type(model).__dict__
    assert "similarity" not in type(model).__dict__
    assert "similarity_pairwise" not in type(model).__dict__
    a = np.array([[3, 4], [-4, 3]], dtype=np.float32)
    b = np.array([[4, 3], [3, 4]], dtype=np.float32)
    expected = (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (
        b / np.linalg.norm(b, axis=1, keepdims=True)
    ).T
    np.testing.assert_allclose(np.asarray(model.similarity(a, b)), expected, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(model.similarity_pairwise(a, b)), np.diag(expected), atol=1e-6
    )


@pytest.mark.parametrize("device", [None, "cuda", "cuda:0"])
def test_device_default(checkpoints, device):
    model = encoder_module.PresonanceEmotionEncoder(
        checkpoint_dir=checkpoints, device=device
    )
    assert model.device == (device or "cuda")


def test_invalid_checkpoint_order(checkpoints):
    path = checkpoints / "config.json"
    data = json.loads(path.read_text())
    data["models"].reverse()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unexpected model order"):
        encoder_module.PresonanceEmotionEncoder(checkpoint_dir=checkpoints)


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.5, "2", None])
def test_invalid_batch_sizes_fail_before_loading(checkpoints, batch_size):
    model = encoder_module.PresonanceEmotionEncoder(checkpoint_dir=checkpoints)
    with pytest.raises(ValueError, match="positive integer"):
        model.predict_features([], batch_size=batch_size)


def test_cpu_request_is_explicitly_unsupported(checkpoints):
    model = encoder_module.PresonanceEmotionEncoder(
        checkpoint_dir=checkpoints, device="cpu"
    )
    with pytest.raises(ValueError, match="requires CUDA"):
        model.predict_features([], batch_size=1)


@pytest.mark.parametrize("bad", [[[0, 0]], [[np.nan, 1]], [[np.inf, 1]]])
def test_nonfinite_or_zero_features_fail(bad):
    with pytest.raises(ValueError, match="finite, nonzero"):
        encoder_module.unit(bad)


def test_combine_requires_all_four_blocks():
    with pytest.raises(ValueError, match="four feature blocks"):
        encoder_module.combine([np.ones((1, 2))] * 3)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 64])
def test_encode_forwards_requested_batch_size(checkpoints, batch_size):
    model = encoder_module.PresonanceEmotionEncoder(checkpoint_dir=checkpoints)
    marker = np.ones((3, 8), dtype=np.float32)
    model.predict_features = Mock(return_value=marker)
    assert (
        model.encode([{"image": [1, 2]}, {"image": [3]}], batch_size=batch_size)
        is marker
    )
    model.predict_features.assert_called_once_with([1, 2, 3], batch_size=batch_size)


def test_encode_preserves_published_default(checkpoints):
    model = encoder_module.PresonanceEmotionEncoder(checkpoint_dir=checkpoints)
    model.predict_features = Mock(return_value=np.ones((1, 8), dtype=np.float32))
    model.encode([{"image": [1]}])
    model.predict_features.assert_called_once_with([1], batch_size=64)


@pytest.fixture
def synthetic_inference(monkeypatch):
    import torch
    from PIL import Image

    calls = []
    timm = types.ModuleType("timm")
    data = types.ModuleType("timm.data")
    tensors = types.ModuleType("safetensors.torch")

    class Model:
        def __init__(self, architecture):
            self.index = int(architecture.rsplit("_", 1)[1])

        def load_state_dict(self, state, strict):
            assert strict

        def eval(self):
            return self

        def requires_grad_(self, value):
            assert not value
            return self

        def to(self, device):
            assert device.startswith("cuda")
            return self

    def make_model(architecture, **kwargs: Any):
        assert kwargs == {"pretrained": False, "num_classes": 8}
        return Model(architecture)

    def features(model, batch):
        calls.append((model.index, len(batch)))
        value = batch[:, 0, 0, 0].numpy()
        return np.stack([value + 1, value + model.index + 2], axis=1)

    def transform_factory(**kwargs: Any):
        assert kwargs == {"is_training": False}
        return lambda image: torch.as_tensor(np.asarray(image).copy()).permute(2, 0, 1)

    timm.create_model = make_model
    data.create_transform = transform_factory
    tensors.load_file = lambda path: {"weight": torch.tensor([1.0])}
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setitem(sys.modules, "timm.data", data)
    monkeypatch.setitem(sys.modules, "safetensors.torch", tensors)
    original_to = torch.Tensor.to

    def cpu_test_to(tensor, *args: Any, **kwargs: Any):
        if args and isinstance(args[0], str) and args[0].startswith("cuda"):
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_test_to)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(encoder_module, "feature_batch", features)
    images = [Image.new("RGB", (2, 2), (index, 0, 0)) for index in range(65)]
    return calls, images, features


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 32, 64, 65])
def test_requested_batches_preserve_order_and_four_block_math(
    checkpoints, synthetic_inference, batch_size
):
    calls, images, _ = synthetic_inference
    model = encoder_module.PresonanceEmotionEncoder(checkpoint_dir=checkpoints)
    actual = model.predict_features(images, batch_size=batch_size)
    expected = []
    for index in range(4):
        raw = np.array(
            [[value + 1, value + index + 2] for value in range(65)], dtype=np.float32
        )
        expected.append(raw / np.linalg.norm(raw, axis=1, keepdims=True))
    expected = np.concatenate(expected, axis=1) * np.float32(0.5)
    np.testing.assert_array_equal(actual, expected)
    sizes = [min(batch_size, 65 - offset) for offset in range(0, 65, batch_size)]
    assert calls == [(index, size) for index in range(4) for size in sizes]


def test_corrupt_checkpoint_stops_before_model_loading(
    checkpoints, synthetic_inference
):
    calls, images, _ = synthetic_inference
    (checkpoints / "model_0.safetensors").write_bytes(b"corrupt")
    model = encoder_module.PresonanceEmotionEncoder(checkpoint_dir=checkpoints)
    with pytest.raises(ValueError, match="Checkpoint SHA256 mismatch"):
        model.predict_features(images, batch_size=2)
    assert not calls
