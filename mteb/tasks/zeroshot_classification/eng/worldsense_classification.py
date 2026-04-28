from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification

CITATION = r"""
@inproceedings{hong2025worldsense,
  author = {Hong, Jack and Yan, Shilin and Cai, Jiayin and Jiang, Xiaolong and Hu, Yao and Xie, Weidi},
  journal = {arXiv preprint arXiv:2502.04326},
  title = {Worldsense: Evaluating real-world omnimodal understanding for multimodal llms},
  year = {2025},
}
"""


class WorldSenseAudioVideoZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="WorldSenseAudioVideoZeroShot",
        description="WorldSense is a multimodal video understanding benchmark encompassing visual, audio, and text inputs. Videos are categorized into 8 primary domains across 67 fine-grained subcategories. This zero-shot classification task predicts the domain category of a video clip",
        reference="https://arxiv.org/abs/2502.04326",
        dataset={
            "path": "mteb/WorldSense_1min",
            "revision": "10c7ce0eb32d620f1f685bfedde2724066068a1c",
        },
        type="VideoZeroshotClassification",
        category="va2t",
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
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name = ("video", "audio")
    label_column_name: str = "domain"

    def get_candidate_labels(self) -> list[str]:
        return [
            name for name in self.dataset["test"].features[self.label_column_name].names
        ]


class WorldSenseVideoZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="WorldSenseVideoZeroShot",
        description="WorldSense is a multimodal video understanding benchmark encompassing visual, audio, and text inputs. Videos are categorized into 8 primary domains across 67 fine-grained subcategories. This zero-shot classification task predicts the domain category of a video clip",
        reference="https://arxiv.org/abs/2502.04326",
        dataset={
            "path": "mteb/WorldSense_1min",
            "revision": "10c7ce0eb32d620f1f685bfedde2724066068a1c",
        },
        type="VideoZeroshotClassification",
        category="v2t",
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
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name = "video"
    label_column_name: str = "domain"

    def get_candidate_labels(self) -> list[str]:
        return [
            name for name in self.dataset["test"].features[self.label_column_name].names
        ]