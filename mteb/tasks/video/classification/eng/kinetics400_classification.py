import datasets
from datasets import DatasetDict, Features, load_dataset

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types._encoder_io import VideoInputItem


class Kinetics400Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Kinetics400",
        description="Kinetics-400 is a large-scale action recognition dataset containing 400 human action classes from YouTube videos. Each clip is approximately 10 seconds long.",
        reference="https://arxiv.org/abs/1705.06950",
        dataset={
            "path": "mteb/kinetics-400",
            "revision": "e5b93b6eae80b8c9e9c88a381baae84d29b34fd2",
        },
        type="VideoClassification",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2017-05-22",
            "2017-05-22",
        ),
        domains=["Scene"],
        task_subtypes=["Activity recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=r"""
@article{kay2017kinetics,
  author = {Will Kay and Joao Carreira and Karen Simonyan and Brian Zhang and Chloe Hillier and Sudheendra Vijayanarasimhan and Fabio Viola and Tim Green and Trevor Back and Paul Natsev and Mustafa Suleyman and Andrew Zisserman},
  eprint = {1705.06950},
  journal = {CoRR},
  title = {The Kinetics Human Action Video Dataset},
  url = {https://arxiv.org/abs/1705.06950},
  volume = {abs/1705.06950},
  year = {2017},
}
""",
    )

    input_column_name: str = "video"
    label_column_name: str = "label"

    is_cross_validation: bool = False

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return

        dataset = load_dataset(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
        )

        def _combine_modalities(example: dict) -> dict:
            example["video"] = [
                VideoInputItem(
                    frames=example["video"],
                    audio=example.pop("audio"),
                )
            ]
            return example

        merged = {}
        for split_name, split in dataset.items():
            split_features = split.features
            merged[split_name] = split.map(
                _combine_modalities,
                features=Features(
                    {
                        k: v
                        for k, v in split_features.items()
                        if k != "audio"
                    }
                    | {
                        "video": datasets.List(
                            feature={
                                "frames": split_features["video"],
                                "audio": split_features["audio"],
                            }
                        ),
                    }
                ),
                writer_batch_size=50,
            )

        self.dataset = DatasetDict(merged)
        self.data_loaded = True
