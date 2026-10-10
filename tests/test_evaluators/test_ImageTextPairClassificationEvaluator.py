"""Tests for `mteb._evaluators.image.imagetext_pairclassification_evaluator`."""

from __future__ import annotations


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
