from __future__ import annotations

from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class BreakfastClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="BreakfastClustering",
        description=(
            "Clustering of video clips into 10 breakfast-related activity categories "
            "from the Breakfast Actions dataset (433 test videos from camera-01, "
            "capped at 50 per class)."
        ),
        reference="https://ieeexplore.ieee.org/document/6909500",
        dataset={
            "path": "mteb/Breakfast",
            "revision": "59a874899eb241993794a3454c37829727c3b559",
        },
        type="VideoClustering",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2014-06-23", "2014-06-28"),
        domains=["Activity", "Instructional"],
        task_subtypes=["Activity recognition"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kuehne2014language,
  author = {Kuehne, Hilde and Arslan, Ali and Serre, Thomas},
  booktitle = {2014 IEEE Conference on Computer Vision and Pattern Recognition},
  doi = {10.1109/CVPR.2014.338},
  pages = {3325-3332},
  title = {The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities},
  year = {2014},
}
""",
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
