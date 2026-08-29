from collections import Counter, defaultdict

import pytest
from datasets import Dataset

from mteb.tasks.pair_classification.zxx import moving_fashion_pc
from mteb.tasks.pair_classification.zxx.moving_fashion_pc import (
    MovingFashionV2IPairClassification,
    build_balanced_pair_manifest,
)


def test_build_balanced_pair_manifest() -> None:
    video_ids = [f"videos/v{i}.mp4" for i in range(6)]
    image_ids = [
        "imgs/h1.png",
        "imgs/h2.png",
        "imgs/h3.png",
        "imgs/h4.png",
        "imgs/r1.jpg",
        "imgs/r2.jpg",
    ]
    qrel_video_ids = video_ids
    qrel_image_ids = [
        "imgs/h1.png",
        "imgs/h2.png",
        "imgs/h3.png",
        "imgs/r1.jpg",
        "imgs/r2.jpg",
        "imgs/r1.jpg",
    ]

    manifest = build_balanced_pair_manifest(
        video_ids, image_ids, qrel_video_ids, qrel_image_ids
    )
    assert manifest == build_balanced_pair_manifest(
        video_ids, image_ids, qrel_video_ids, qrel_image_ids
    )
    assert Counter(manifest["label"]) == {0: 6, 1: 6}

    positives_by_video: dict[str, set[str]] = defaultdict(set)
    for video_id, image_id in zip(qrel_video_ids, qrel_image_ids, strict=True):
        positives_by_video[video_id].add(image_id)

    negative_usage: Counter[str] = Counter()
    for video_id, image_id, label, source in zip(
        manifest["video_id"],
        manifest["image_id"],
        manifest["label"],
        manifest["source_subset"],
        strict=True,
    ):
        expected_source = "hard" if image_id.endswith(".png") else "regular"
        assert source == expected_source
        if label == 0:
            assert image_id not in positives_by_video[video_id]
            negative_usage[image_id] += 1

    for source_suffix in (".png", ".jpg"):
        source_usage = [
            negative_usage[image_id]
            for image_id in image_ids
            if image_id.endswith(source_suffix)
        ]
        assert max(source_usage) - min(source_usage) <= 1


def test_build_balanced_pair_manifest_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="source group"):
        build_balanced_pair_manifest(
            ["videos/v1.mp4"],
            ["imgs/i1.webp"],
            ["videos/v1.mp4"],
            ["imgs/i1.webp"],
        )


def test_task_load_data_reuses_retrieval_configs(monkeypatch) -> None:
    datasets = {
        "queries": Dataset.from_dict(
            {
                "id": ["videos/v1.mp4", "videos/v2.mp4"],
                "video": ["video-1", "video-2"],
            }
        ),
        "corpus": Dataset.from_dict(
            {
                "id": ["imgs/i1.png", "imgs/i2.png"],
                "image": ["image-1", "image-2"],
            }
        ),
        "qrels": Dataset.from_dict(
            {
                "query-id": ["videos/v1.mp4", "videos/v2.mp4"],
                "corpus-id": ["imgs/i1.png", "imgs/i2.png"],
                "score": [1, 1],
            }
        ),
    }

    def fake_load_dataset(path, config, **kwargs):
        assert path == "pranitchawla/MovingFashion"
        assert kwargs["split"] == "test"
        return datasets[config]

    monkeypatch.setattr(moving_fashion_pc, "load_dataset", fake_load_dataset)
    task = MovingFashionV2IPairClassification()
    task.load_data()

    assert len(task.dataset["test"]) == 4
    assert Counter(task.dataset["test"]["label"]) == {0: 2, 1: 2}
    assert set(task.dataset["test"].column_names) == {
        "video_id",
        "image_id",
        "label",
        "source_subset",
        "video",
        "image",
    }
