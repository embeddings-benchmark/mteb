"""Tests for datasets whose samples have interleaved modality coverage.

Such a dataset declares every modality it uses, but an individual row only carries
the ones it actually has — some rows text-only, some image-only, some both.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest
import torch
from datasets import Dataset

import mteb
from mteb._create_dataloaders import _corpus_to_dict, create_dataloader
from mteb.abstasks._statistics_calculation import (
    MISSING_MODALITY_HASH,
    calculate_image_statistics,
    calculate_text_statistics,
    compute_image_hashes,
    compute_text_hashes,
)
from mteb.mocks.mock_tasks import MockAny2AnyRetrievalInterleavedIT2ITTask
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.cache_wrappers.cache_backends._hash_utils import _hash_item
from mteb.models.modality_utils import (
    get_present_indices,
    is_interleaved,
    is_modality_present,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType


@pytest.fixture
def images():
    from PIL import Image

    rng = np.random.default_rng(42)
    return [
        Image.fromarray(rng.integers(0, 255, (8, 8, 3)).astype("uint8"))
        for _ in range(2)
    ]


def _task_metadata(category: str, modalities: list[str]):
    task = MockAny2AnyRetrievalInterleavedIT2ITTask()
    metadata = task.metadata.model_copy(deep=True)
    metadata.category = category
    metadata.modalities = modalities
    return metadata


def test_interleaved_corpus_dataloader(images):
    """An absent text reaches the model as "", an absent image as None."""
    corpus = Dataset.from_dict(
        {
            "id": ["d0", "d1", "d2"],
            "text": ["a document", None, "another document"],
            "image": [None, images[0], images[1]],
        }
    )
    loader = create_dataloader(
        corpus,
        task_metadata=_task_metadata("it2it", ["image", "text"]),
        prompt_type=PromptType.document,
        batch_size=3,
    )
    (batch,) = list(loader)

    assert batch["text"] == ["a document", "", "another document"]
    assert batch["image"][0] is None
    assert batch["image"][1] is not None and batch["image"][2] is not None


def test_interleaved_queries_dataloader(images):
    queries = Dataset.from_dict(
        {
            "id": ["q0", "q1"],
            "text": [None, "a query"],
            "image": [images[0], None],
        }
    )
    loader = create_dataloader(
        queries,
        task_metadata=_task_metadata("it2it", ["image", "text"]),
        prompt_type=PromptType.query,
        batch_size=2,
    )
    (batch,) = list(loader)

    assert batch["text"] == ["", "a query"]
    assert batch["query"] == ["", "a query"]
    assert batch["image"][1] is None


def test_conversation_is_detected_from_the_first_row_that_has_text():
    """An empty conversation is still a conversation, and a text-less row is skipped."""
    queries = Dataset.from_dict(
        {
            "id": ["q0", "q1", "q2"],
            "text": [
                None,  # carries no text at all
                [],  # an empty conversation history is still a list
                [{"role": "user", "content": "a turn"}],
            ],
        }
    )
    loader = create_dataloader(
        queries,
        task_metadata=_task_metadata("t2t", ["text"]),
        prompt_type=PromptType.query,
        batch_size=3,
    )
    (batch,) = list(loader)

    assert "conversation" in batch, "should route through the conversation path"
    assert batch["conversation"][2] == [{"role": "user", "content": "a turn"}]
    assert batch["text"] == ["", "", "user: a turn"]


def test_conversation_detection_when_every_history_is_empty():
    """An all-empty conversation column is still a conversation column.

    Guards the detection against a truthiness check, which would see nothing at all
    here and misroute the dataset through the plain-text query path.
    """
    queries = Dataset.from_dict({"id": ["q0", "q1"], "text": [None, []]})
    loader = create_dataloader(
        queries,
        task_metadata=_task_metadata("t2t", ["text"]),
        prompt_type=PromptType.query,
        batch_size=2,
    )
    (batch,) = list(loader)

    assert "conversation" in batch
    assert batch["conversation"] == [[], []]
    assert batch["text"] == ["", ""]


def test_missing_value_outside_an_input_column_still_raises():
    """A None in `id` is a broken dataset, not an interleaved one."""
    dataset = Dataset.from_dict(
        {"id": ["d0", None], "text": ["a document", "another document"]}
    )
    loader = create_dataloader(
        dataset,
        task_metadata=_task_metadata("t2t", ["text"]),
        prompt_type=PromptType.document,
        batch_size=2,
    )
    with pytest.raises(ValueError, match="Found None in batch for key 'id'"):
        list(loader)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        # unchanged for every row that carries text
        ({"id": "1", "text": "x"}, {"id": "1", "text": "x", "body": "x"}),
        ({"id": "1", "text": "  x  "}, {"id": "1", "text": "x", "body": "  x  "}),
        (
            {"id": "1", "text": "x", "title": "T"},
            {"id": "1", "text": "T x", "body": "x", "title": "T"},
        ),
        ({"id": "1", "text": ""}, {"id": "1", "text": "", "body": ""}),
        # absence is preserved, so the statistics can tell it from an empty string
        ({"id": "1", "text": None}, {"id": "1", "text": None, "body": None}),
        (
            {"id": "1", "text": None, "title": "T"},
            {"id": "1", "text": "T", "body": None, "title": "T"},
        ),
    ],
)
def test_corpus_to_dict(row: dict[str, Any], expected: dict[str, Any]):
    assert _corpus_to_dict(row) == expected


def test_text_statistics_exclude_absent_rows():
    stats = calculate_text_statistics(["abcd", None, "ef"])

    assert stats["total_text_length"] == 6
    assert stats["min_text_length"] == 2
    assert stats["max_text_length"] == 4
    assert stats["average_text_length"] == 3.0
    assert stats["unique_texts"] == 2


def test_empty_string_still_counts_as_present():
    """Only None means absent — an empty text is a text of length 0."""
    assert calculate_text_statistics(["abcd", "", "ef"])["min_text_length"] == 0


def test_image_statistics_exclude_absent_rows(images):
    stats = calculate_image_statistics([images[0], None, images[1]])

    assert stats["unique_images"] == 2
    assert stats["min_image_width"] == 8


def test_statistics_of_a_fully_absent_modality():
    assert calculate_text_statistics([None, None]) == {
        "total_text_length": 0,
        "min_text_length": 0,
        "average_text_length": 0.0,
        "max_text_length": 0,
        "unique_texts": 0,
    }


def test_hashes_stay_row_aligned(images):
    """Hashes are zipped positionally elsewhere, so absent rows get a sentinel."""
    image_hashes = compute_image_hashes([images[0], None])
    assert len(image_hashes) == 2
    assert image_hashes[1] == MISSING_MODALITY_HASH

    assert compute_text_hashes(["a", None]) == ["a", MISSING_MODALITY_HASH]


def test_cache_key_of_an_existing_item_is_unchanged():
    """Existing on-disk caches must stay valid, so present values hash as before."""
    assert (
        _hash_item({"text": "a document"}) == hashlib.sha256(b"a document").hexdigest()
    )


def test_cache_key_ignores_absent_modalities(images):
    """An absent modality contributes nothing, exactly like an absent column."""
    assert _hash_item({"text": "a document", "image": None}) == _hash_item(
        {"text": "a document"}
    )
    assert _hash_item({"text": None, "image": images[0]}) == _hash_item(
        {"image": images[0]}
    )

    with pytest.raises(TypeError):
        _hash_item({"text": None, "image": None})


def test_is_modality_present():
    assert is_modality_present("a text")
    assert not is_modality_present("")
    assert not is_modality_present(None)
    assert not is_modality_present([])


def test_present_indices_and_interleaving():
    batch = {"text": ["a", "", "c"], "image": [None, 1, None]}

    assert get_present_indices(batch, "text") == [0, 2]
    assert get_present_indices(batch, "image") == [1]
    assert get_present_indices(batch, "audio") == []

    assert is_interleaved(batch, "text")
    assert not is_interleaved({"text": ["a", "b"]}, "text")
    assert not is_interleaved(batch, "audio")


class _InterleavedEncoder(AbsEncoder):
    """Embeds each row from whichever modalities it carries, and records what it saw."""

    mteb_model_meta = ModelMeta(
        loader=None,
        name="mock/interleaved",
        model_type=["dense"],
        languages=["eng-Latn"],
        revision="1",
        release_date="2024-01-01",
        modalities=["image", "text"],
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=4,
        license=None,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        framework=["PyTorch"],
        reference=None,
        similarity_fn_name=ScoringFunction.COSINE,
        use_instructions=False,
        training_datasets=None,
    )

    def __init__(self) -> None:
        self.seen_batches: list[BatchedInput] = []

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        embeddings = []
        for batch in inputs:
            self.seen_batches.append(batch)
            for text, image in zip(batch["text"], batch["image"], strict=True):
                vector = np.zeros(4, dtype=np.float32)
                if is_modality_present(text):
                    vector += np.full(4, len(text), dtype=np.float32)
                if is_modality_present(image):
                    vector += np.full(4, image.size[0], dtype=np.float32)
                embeddings.append(vector)
        return np.vstack(embeddings)


def test_evaluate_a_task_with_interleaved_modalities():
    model = _InterleavedEncoder()
    task = MockAny2AnyRetrievalInterleavedIT2ITTask()

    results = mteb.evaluate(model, task, cache=None, co2_tracker=False)

    assert len(results) == 1
    assert results[0].get_score() > 0

    # the model was handed every row, with the absent modality marked rather than dropped
    texts = [text for batch in model.seen_batches for text in batch["text"]]
    images = [image for batch in model.seen_batches for image in batch["image"]]
    assert "" in texts, "an absent text should reach the model as an empty string"
    assert None in images, "an absent image should reach the model as None"
    assert len(texts) == len(images) == 6, "queries and documents, three rows each"


def test_descriptive_statistics_of_an_interleaved_task():
    task = MockAny2AnyRetrievalInterleavedIT2ITTask()
    stats = task.calculate_descriptive_statistics(overwrite_results=True)
    task.metadata.descriptive_stat_path.unlink()

    assert stats["test"] == task.expected_stats["test"]


def _stub_clip_model():
    """A CLIPModel whose towers embed a text by its length and an image by its width."""
    from mteb.models.model_implementations.clip_models import CLIPModel

    model = CLIPModel.__new__(CLIPModel)
    model._encode_texts = lambda texts: torch.tensor(
        [[float(len(text))] * 4 for text in texts]
    )
    model._encode_images = lambda images: torch.tensor(
        [[float(image.size[0])] * 4 for image in images]
    )
    return model


def _loader(rows: list[dict[str, Any]], batch_size: int = 2):
    return create_dataloader(
        Dataset.from_list(rows),
        task_metadata=_task_metadata("it2it", ["image", "text"]),
        prompt_type=None,
        batch_size=batch_size,
    )


def test_clip_fusion_is_unchanged_when_every_row_has_both(images):
    """The fused path must still be text + image for a fully populated dataset."""
    model = _stub_clip_model()
    rows = [
        {"id": "d0", "text": "abc", "image": images[0]},
        {"id": "d1", "text": "de", "image": images[1]},
    ]

    fused = model.get_fused_embeddings(_loader(rows), show_progress_bar=False)
    separate = model.get_text_embeddings(
        _loader(rows), show_progress_bar=False
    ) + model.get_image_embeddings(_loader(rows), show_progress_bar=False)

    assert torch.equal(fused, separate)


def test_clip_fusion_uses_only_the_modalities_a_row_carries(images):
    model = _stub_clip_model()
    rows = [
        {"id": "d0", "text": "abc", "image": None},
        {"id": "d1", "text": None, "image": images[0]},
        {"id": "d2", "text": "de", "image": images[1]},
    ]

    fused = model.get_fused_embeddings(
        _loader(rows, batch_size=3), show_progress_bar=False
    )

    # text-only row: len("abc"); image-only row: width 8; both: 2 + 8
    assert fused[:, 0].tolist() == [3.0, 8.0, 10.0]
