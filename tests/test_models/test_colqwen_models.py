from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import Mock

from mteb.models.model_implementations.colqwen_models import ColQwen3_5Wrapper


def test_colqwen3_5_wrapper_forwards_revision(monkeypatch):
    revision = "registered-revision"
    model_class = Mock()
    model_class.from_pretrained.return_value = Mock()
    processor_class = Mock()

    models_module = ModuleType("colpali_engine.models")
    models_module.ColQwen3_5 = model_class
    models_module.ColQwen3_5Processor = processor_class
    engine_module = ModuleType("colpali_engine")
    engine_module.models = models_module
    monkeypatch.setitem(sys.modules, "colpali_engine", engine_module)
    monkeypatch.setitem(sys.modules, "colpali_engine.models", models_module)

    ColQwen3_5Wrapper("org/model", revision=revision, device="cpu")

    model_class.from_pretrained.assert_called_once()
    model_kwargs = model_class.from_pretrained.call_args.kwargs
    assert model_kwargs["revision"] == revision
    assert model_kwargs["adapter_kwargs"]["revision"] == revision

    processor_class.from_pretrained.assert_called_once()
    processor_kwargs = processor_class.from_pretrained.call_args.kwargs
    assert processor_kwargs["revision"] == revision
