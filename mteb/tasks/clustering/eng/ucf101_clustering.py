from __future__ import annotations

from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

BIBTEX = r"""
@misc{Soomro2012UCF101,
  archiveprefix = {arXiv},
  author = {Soomro, Khurram and Zamir, Amir Roshan and Shah, Mubarak},
  eprint = {1212.0402},
  primaryclass = {cs.CV},
  title = {UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  url = {https://arxiv.org/abs/1212.0402},
  year = {2012},
}
"""

DATASET = {
    "path": "mteb/UCF101-51VA",
    "revision": "866b006d84629d66d9927646db89bd43381925e7",
}

DESCRIPTION_BASE = (
    "Clustering of video clips into 51 human action categories from the UCF101 dataset."
)


class UCF101AudioVideoClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="UCF101AudioVideoClustering",
        description=DESCRIPTION_BASE + " Uses synchronized video and audio.",
        reference="https://arxiv.org/abs/1212.0402",
        dataset=DATASET,
        type="VideoClustering",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2012-01-01", "2012-12-03"),
        domains=["Web", "Scene"],
        task_subtypes=["Activity recognition"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name = ("video", "audio")
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            self.dataset[split] = self.dataset[split].select_columns(
                ["video", "audio", "label"],
            )


class UCF101VideoClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="UCF101VideoClustering",
        description=DESCRIPTION_BASE + " Uses video only.",
        reference="https://arxiv.org/abs/1212.0402",
        dataset=DATASET,
        type="VideoClustering",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2012-01-01", "2012-12-03"),
        domains=["Web", "Scene"],
        task_subtypes=["Activity recognition"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "video"
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            self.dataset[split] = self.dataset[split].select_columns(
                ["video", "label"],
            )
