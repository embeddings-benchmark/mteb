from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset, Features, Value
from datasets import Image as HFImage
from PIL import Image

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_implementations.random_baseline import (
    _batch_to_embeddings,
    _image_to_vector,
    _string_to_vector,
)
from mteb.models.sentence_transformer_wrapper import _batch_to_modality_dicts
from mteb.tasks.retrieval.eng.mixbench_retrieval import MixBenchMSCOCO
from mteb.types import PromptType


def _image(color: str) -> Image.Image:
    return Image.new("RGB", (2, 2), color=color)


def _mixed_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "id": ["text", "image", "both"],
            "text": ["text only", "", "both"],
            "image": [None, _image("red"), _image("blue")],
        },
        features=Features(
            {"id": Value("string"), "text": Value("string"), "image": HFImage()}
        ),
    )


def test_per_sample_modalities_omit_missing_values():
    red = _image("red")
    blue = _image("blue")
    batch = {
        "text": ["text only", "", "both"],
        "image": [None, red, blue],
    }

    samples = _batch_to_modality_dicts(batch, ["text", "image"])

    assert samples == [
        {"text": "text only"},
        {"image": red},
        {"text": "both", "image": blue},
    ]


def test_per_sample_modalities_reject_empty_input():
    with pytest.raises(ValueError, match="without any populated modality"):
        _batch_to_modality_dicts({"text": [""], "image": [None]}, ["text", "image"])


def test_instruct_wrapper_preserves_present_modalities_with_incomplete_metadata():
    image = _image("red")

    class Inputs:
        dataset = SimpleNamespace(features={"text": object(), "image": object()})

        def __iter__(self):
            yield {"text": ["dense input"], "image": [image]}

    class RecordingModel:
        encoded_inputs = None

        def encode(self, inputs, **kwargs):
            self.encoded_inputs = inputs
            return np.zeros((len(inputs), 2), dtype=np.float32)

    wrapper = object.__new__(InstructSentenceTransformerModel)
    wrapper.apply_instruction_to_passages = True
    wrapper.mteb_model_meta = SimpleNamespace(modalities=["text"])
    wrapper.get_task_instruction = lambda *_: None
    wrapper.model = RecordingModel()

    embeddings = wrapper.encode(
        Inputs(),
        task_metadata=SimpleNamespace(name="dense-regression"),
        hf_split="test",
        hf_subset="default",
        prompt_type=PromptType.query,
    )

    assert embeddings.shape == (1, 2)
    assert wrapper.model.encoded_inputs is not None
    assert wrapper.model.encoded_inputs[0]["text"] == "dense input"
    assert wrapper.model.encoded_inputs[0]["image"] is image


def test_random_baseline_combines_only_present_modalities():
    red = _image("red")
    blue = _image("blue")
    embeddings = _batch_to_embeddings(
        [
            {
                "text": ["text only", "", "both"],
                "image": [None, red, blue],
            }
        ],
        embedding_dim=8,
    )

    np.testing.assert_allclose(embeddings[0], _string_to_vector("text only", 8))
    np.testing.assert_allclose(embeddings[1], _image_to_vector(red, 8))
    np.testing.assert_allclose(
        embeddings[2],
        np.mean(
            [_string_to_vector("both", 8), _image_to_vector(blue, 8)],
            axis=0,
        ),
    )


def test_mixbench_loader_normalizes_ids_and_missing_text(monkeypatch):
    import mteb.tasks.retrieval.eng.mixbench_retrieval as mixbench

    datasets = {
        "queries": Dataset.from_dict({"id": [1], "text": ["query"], "image": [None]}),
        "mixed_corpus": Dataset.from_dict({"id": [1], "text": [None], "image": [None]}),
    }

    loaded_splits = []

    def fake_load_split(config, split):
        loaded_splits.append((config, split))
        return datasets[split]

    monkeypatch.setattr(mixbench, "_load_split", fake_load_split)
    monkeypatch.setattr(mixbench, "_load_qrels", lambda config: {"1": {"1": 1}})
    task = MixBenchMSCOCO()

    task.load_data()

    split = task.dataset["default"]["test"]
    assert split["queries"]["id"] == ["1"]
    assert split["corpus"]["id"] == ["1"]
    assert split["corpus"]["text"] == [""]
    assert split["relevant_docs"] == {"1": {"1": 1}}
    assert loaded_splits == [("MSCOCO", "queries"), ("MSCOCO", "mixed_corpus")]


def test_retrieval_statistics_ignore_absent_modalities():
    task = MixBenchMSCOCO()
    task.dataset = {
        "default": {
            "test": {
                "queries": Dataset.from_dict(
                    {"id": ["q"], "text": ["query"], "image": [None]},
                    features=Features(
                        {
                            "id": Value("string"),
                            "text": Value("string"),
                            "image": HFImage(),
                        }
                    ),
                ),
                "corpus": _mixed_dataset(),
                "relevant_docs": {"q": {"both": 1}},
                "top_ranked": None,
            }
        }
    }
    task.data_loaded = True

    stats = task._calculate_descriptive_statistics_from_split("test")

    assert stats["num_documents_with_text"] == 2
    assert stats["num_documents_with_image"] == 2
    assert stats["num_queries_with_text"] == 1
    assert stats["num_queries_with_image"] == 0
    assert stats["documents_text_statistics"]["unique_texts"] == 2
    assert stats["documents_image_statistics"]["unique_images"] == 2
