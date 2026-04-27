from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class WorldSenseClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WorldSenseClassification",
        description="WorldSense is a multimodal video understanding benchmark encompassing visual, audio, and text inputs. Videos are categorized into 8 primary domains across 67 fine-grained subcategories. This classification task predicts the domain category of a video clip",
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
        domains=["Scene", "AudioScene", "Music", "Entertainment"],
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


class WorldSenseVideoClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WorldSenseVideoClassification",
        description="WorldSense is a multimodal video understanding benchmark encompassing visual, audio, and text inputs. Videos are categorized into 8 primary domains across 67 fine-grained subcategories. This classification task predicts the domain category of a video clip",
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
        domains=["Scene", "AudioScene", "Music", "Entertainment"],
        task_subtypes=["Scene recognition"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["video"],
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
    input_column_name = "video"
    label_column_name: str = "domain"
    is_cross_validation: bool = True
    train_split: str = "test"
