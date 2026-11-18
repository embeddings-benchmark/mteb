"""Tests for `mteb._evaluators.image.imagetext_pairclassification_evaluator`."""

from __future__ import annotations

from typing import Any


def test_image_dataset_works_in_dataloader_worker_processes() -> None:
    """With `num_proc > 1` the image dataset and collate fn are pickled into worker processes.

    `spawn` (the default on macOS and Windows) cannot pickle anything defined inside a function.
    """
    from PIL import Image
    from torch.utils.data import DataLoader

    from mteb._evaluators.image.imagetext_pairclassification_evaluator import (
        _build_image_dataset,
        _image_collate_fn,
    )

    loader = DataLoader(
        _build_image_dataset([Image.new("RGB", (8, 8)) for _ in range(3)]),
        batch_size=2,
        num_workers=2,
        collate_fn=_image_collate_fn,
        multiprocessing_context="spawn",
    )
    assert [len(batch["image"]) for batch in loader] == [2, 1]


def test_images_are_encoded_with_the_batch_size() -> None:
    """Images are batched by `encode_kwargs["batch_size"]`, like the texts, not one at a time."""
    import numpy as np
    from datasets import Dataset
    from PIL import Image

    from mteb._evaluators.image.imagetext_pairclassification_evaluator import (
        ImageTextPairClassificationEvaluator,
    )
    from mteb.mocks.mock_tasks import MockImageTextPairClassificationTask
    from mteb.timing import TimingStack

    batch_sizes: dict[str, list[int]] = {"image": [], "text": []}

    class RecordingEncoder:
        def encode(self, inputs, **kwargs: Any):
            n_inputs = 0
            for batch in inputs:
                modality = "image" if "image" in batch else "text"
                batch_sizes[modality].append(len(batch[modality]))
                n_inputs += len(batch[modality])
            return np.ones((n_inputs, 4))

    evaluator = ImageTextPairClassificationEvaluator(
        Dataset.from_dict(
            {
                "image": [Image.new("RGB", (8, 8)) for _ in range(3)],
                "caption": ["a red square", "a blue square", "a green square"],
            }
        ),
        images_column_names="image",
        texts_column_names="caption",
        num_images_per_sample=1,
        num_texts_per_sample=1,
        task_metadata=MockImageTextPairClassificationTask.metadata,
        hf_split="test",
        hf_subset="default",
        timer=TimingStack(),
    )
    evaluator(RecordingEncoder(), encode_kwargs={"batch_size": 2})

    assert batch_sizes == {"image": [2, 1], "text": [2, 1]}
