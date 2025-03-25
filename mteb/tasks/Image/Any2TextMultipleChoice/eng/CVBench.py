from __future__ import annotations

import datasets

from mteb.abstasks.Image.AbsTaskAny2TextMultipleChoice import (
    AbsTaskAny2TextMultipleChoice,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


def transform_choices(example):
    mapping = {"(A)": 0, "(B)": 1, "(C)": 2, "(D)": 3, "(E)": 4, "(F)": 5}
    example["answer"] = mapping[example["answer"]]
    return example


class CVBenchCount(AbsTaskAny2TextMultipleChoice):
    metadata = TaskMetadata(
        name="CVBenchCount",
        description="count the number of objects in the image.",
        reference="https://arxiv.org/pdf/2406.16860",
        dataset={
            "path": "nyu-visionx/CV-Bench",
            "revision": "22409a927ab5cf68e3655023d51694587455fc99",
        },
        type="VisionCentric",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2024-06-24"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{tong2024cambrian,
  title={Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
  author={Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
  journal={arXiv preprint arXiv:2406.16860},
  year={2024}
}""",
        descriptive_stats={
            "n_samples": {"test": 788},
            "avg_character_length": {
                "test": {
                    # to do
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        self.dataset_transform()
        self.dataset = self.dataset.filter(lambda example: example["task"] == "Count")
        self.dataset = self.dataset.map(
            transform_choices,
            remove_columns=[
                "idx",
                "type",
                "filename",
                "source",
                "source_dataset",
                "source_filename",
                "target_class",
                "target_size",
                "bbox",
                "prompt",
            ],
        )
        self.data_loaded = True


class CVBenchRelation(AbsTaskAny2TextMultipleChoice):
    metadata = TaskMetadata(
        name="CVBenchRelation",
        description="decide the relation of the objects in the image.",
        reference="https://arxiv.org/pdf/2406.16860",
        dataset={
            "path": "nyu-visionx/CV-Bench",
            "revision": "22409a927ab5cf68e3655023d51694587455fc99",
        },
        type="VisionCentric",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2024-06-24"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{tong2024cambrian,
  title={Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
  author={Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
  journal={arXiv preprint arXiv:2406.16860},
  year={2024}
}""",
        descriptive_stats={
            "n_samples": {"test": 650},
            "avg_character_length": {
                "test": {
                    # to do
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        self.dataset_transform()
        self.dataset = self.dataset.filter(
            lambda example: example["task"] == "Relation"
        )
        self.dataset = self.dataset.map(
            transform_choices,
            remove_columns=[
                "idx",
                "type",
                "filename",
                "source",
                "source_dataset",
                "source_filename",
                "target_class",
                "target_size",
                "bbox",
                "prompt",
            ],
        )
        self.data_loaded = True


class CVBenchDepth(AbsTaskAny2TextMultipleChoice):
    metadata = TaskMetadata(
        name="CVBenchDepth",
        description="judge the depth of the objects in the image with similarity matching.",
        reference="https://arxiv.org/pdf/2406.16860",
        dataset={
            "path": "nyu-visionx/CV-Bench",
            "revision": "22409a927ab5cf68e3655023d51694587455fc99",
        },
        type="VisionCentric",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2024-06-24"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{tong2024cambrian,
  title={Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
  author={Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
  journal={arXiv preprint arXiv:2406.16860},
  year={2024}
}""",
        descriptive_stats={
            "n_samples": {"test": 600},
            "avg_character_length": {
                "test": {
                    # to do
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        self.dataset_transform()
        self.dataset = self.dataset.filter(lambda example: example["task"] == "Depth")
        self.dataset = self.dataset.map(
            transform_choices,
            remove_columns=[
                "idx",
                "type",
                "filename",
                "source",
                "source_dataset",
                "source_filename",
                "target_class",
                "target_size",
                "bbox",
                "prompt",
            ],
        )
        self.data_loaded = True


class CVBenchDistance(AbsTaskAny2TextMultipleChoice):
    metadata = TaskMetadata(
        name="CVBenchDistance",
        description="judge the distance of the objects in the image with similarity matching.",
        reference="https://arxiv.org/pdf/2406.16860",
        dataset={
            "path": "nyu-visionx/CV-Bench",
            "revision": "22409a927ab5cf68e3655023d51694587455fc99",
        },
        type="VisionCentric",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2024-01-01", "2024-06-24"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{tong2024cambrian,
  title={Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
  author={Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
  journal={arXiv preprint arXiv:2406.16860},
  year={2024}
}""",
        descriptive_stats={
            "n_samples": {"test": 600},
            "avg_character_length": {
                "test": {
                    # to do
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        self.dataset_transform()
        self.dataset = self.dataset.filter(
            lambda example: example["task"] == "Distance"
        )
        self.dataset = self.dataset.map(
            transform_choices,
            remove_columns=[
                "idx",
                "type",
                "filename",
                "source",
                "source_dataset",
                "source_filename",
                "target_class",
                "target_size",
                "bbox",
                "prompt",
            ],
        )
        self.data_loaded = True
