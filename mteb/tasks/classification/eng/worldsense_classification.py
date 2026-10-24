from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset


class WorldSenseAudioVideoClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WorldSenseAudioVideoClassification",
        description="WorldSense is a multimodal video understanding benchmark encompassing visual, audio, and text inputs. Videos are categorized into 8 primary domains across 67 fine-grained subcategories. This classification task predicts the domain category of a video clip Filtered the test split to short (<1min) clips (~1,047 examples), one row per video (568 videos).",
        reference="https://arxiv.org/abs/2502.04326",
        dataset={
            "path": "mteb/WorldSense_1min",
            "revision": "10c7ce0eb32d620f1f685bfedde2724066068a1c",
        },
        type="VideoClassification",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-02-06", "2026-03-01"),
        domains=["AudioScene", "Scene", "Web"],
        task_subtypes=["Scene recognition"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{hong2025worldsense,
  author = {Hong, Jack and Yan, Shilin and Cai, Jiayin and Jiang, Xiaolong and Hu, Yao and Xie, Weidi},
  journal = {arXiv preprint arXiv:2502.04326},
  title = {Worldsense: Evaluating real-world omnimodal understanding for multimodal llms},
  year = {2025},
}
""",
    )
    input_column_name = ("video", "audio")
    label_column_name: str = "domain"
    is_cross_validation: bool = True
    train_split: str = "test"

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.dataset["test"] = _one_row_per_video(self.dataset["test"])


class WorldSenseVideoClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WorldSenseVideoClassification",
        description="WorldSense is a multimodal video understanding benchmark encompassing visual, audio, and text inputs. Videos are categorized into 8 primary domains across 67 fine-grained subcategories. This classification task predicts the domain category of a video clip Filtered the test split to short (<1min) clips (~1,047 examples).",
        reference="https://arxiv.org/abs/2502.04326",
        dataset={
            "path": "mteb/WorldSense_1min",
            "revision": "10c7ce0eb32d620f1f685bfedde2724066068a1c",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-02-06", "2026-03-01"),
        domains=["AudioScene", "Scene", "Web"],
        task_subtypes=["Scene recognition"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        superseded_by="WorldSenseVideoClassification.v2",
        bibtex_citation=r"""
@inproceedings{hong2025worldsense,
  author = {Hong, Jack and Yan, Shilin and Cai, Jiayin and Jiang, Xiaolong and Hu, Yao and Xie, Weidi},
  journal = {arXiv preprint arXiv:2502.04326},
  title = {Worldsense: Evaluating real-world omnimodal understanding for multimodal llms},
  year = {2025},
}
""",
    )
    input_column_name = "video"
    label_column_name: str = "domain"
    is_cross_validation: bool = True
    train_split: str = "test"


class WorldSenseVideoClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WorldSenseVideoClassification.v2",
        description="WorldSense is a multimodal video understanding benchmark encompassing visual, audio, and text inputs. Videos are categorized into 8 primary domains across 67 fine-grained subcategories. This classification task predicts the domain category of a video clip Filtered the test split to short (<1min) clips. Version 2 keeps one row per video (568 videos), so a video's question rows no longer land in both train and test folds.",
        reference="https://arxiv.org/abs/2502.04326",
        dataset={
            "path": "mteb/WorldSense_1min",
            "revision": "10c7ce0eb32d620f1f685bfedde2724066068a1c",
        },
        type="VideoClassification",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-02-06", "2026-03-01"),
        domains=["AudioScene", "Scene", "Web"],
        task_subtypes=["Scene recognition"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        adapted_from=["WorldSenseVideoClassification"],
        bibtex_citation=r"""
@inproceedings{hong2025worldsense,
  author = {Hong, Jack and Yan, Shilin and Cai, Jiayin and Jiang, Xiaolong and Hu, Yao and Xie, Weidi},
  journal = {arXiv preprint arXiv:2502.04326},
  title = {Worldsense: Evaluating real-world omnimodal understanding for multimodal llms},
  year = {2025},
}
""",
    )
    input_column_name = "video"
    label_column_name: str = "domain"
    is_cross_validation: bool = True
    train_split: str = "test"

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        self.dataset["test"] = _one_row_per_video(self.dataset["test"])


def _one_row_per_video(dataset: Dataset) -> Dataset:
    # rows are WorldSense questions, up to 5 per video with the same domain;
    # repeats would put the same video in the train and test folds
    first_rows: dict[str, int] = {}
    for i, video_id in enumerate(dataset["video_id"]):
        first_rows.setdefault(video_id, i)
    return dataset.select(list(first_rows.values()))
