from __future__ import annotations

from datasets import Dataset, DatasetDict

from mteb.abstasks.abstask import AbsTask


def _decoded(ds: Dataset) -> list[str]:
    return [ds.features["label"].int2str(i) for i in ds["label"]]


def test_label_missing_from_one_split_keeps_ids_aligned():
    # "nineties" only appears in train; encoding each split on its own would
    # give "twenties" a different id in train and test
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"label": ["twenties", "nineties", "teens"]}),
            "test": Dataset.from_dict({"label": ["twenties", "teens"]}),
        }
    )
    out = AbsTask.stratified_subsampling(dataset, seed=42, splits=["test"])
    assert out["train"].features["label"] == out["test"].features["label"]
    assert _decoded(out["train"]) == ["twenties", "nineties", "teens"]
    assert _decoded(out["test"]) == ["twenties", "teens"]


def test_matches_class_encode_column_when_splits_share_labels():
    for labels in (["b", "a", "c"], [2, 10, 1]):
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"label": labels}),
                "test": Dataset.from_dict({"label": labels[::-1]}),
            }
        )
        expected = DatasetDict(dataset).class_encode_column("label")
        out = AbsTask.stratified_subsampling(dataset, seed=42, splits=["test"])
        for split in dataset:
            assert out[split]["label"] == expected[split]["label"]
            assert out[split].features["label"] == expected[split].features["label"]
