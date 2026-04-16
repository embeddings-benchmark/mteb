from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)


class Kinetics400ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="Kinetics400ZeroShot",
        description="Kinetics-400 is a large-scale action recognition dataset containing 400 human action classes from YouTube videos. Each clip is approximately 10 seconds long.",
        reference="https://arxiv.org/abs/1705.06950",
        dataset={
            "path": "mteb/kinetics-400",
            "revision": "e5b93b6eae80b8c9e9c88a381baae84d29b34fd2",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2017-05-19",
            "2017-05-19",
        ),
        domains=["Web", "Scene"],
        task_subtypes=["Activity recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{kay2017kineticshumanactionvideo,
  archiveprefix = {arXiv},
  author = {Will Kay and Joao Carreira and Karen Simonyan and Brian Zhang and Chloe Hillier and Sudheendra Vijayanarasimhan and Fabio Viola and Tim Green and Trevor Back and Paul Natsev and Mustafa Suleyman and Andrew Zisserman},
  eprint = {1705.06950},
  primaryclass = {cs.CV},
  title = {The Kinetics Human Action Video Dataset},
  url = {https://arxiv.org/abs/1705.06950},
  year = {2017},
}
""",
    )

    def get_candidate_labels(self) -> list[str]:
        return [
            {name}
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
