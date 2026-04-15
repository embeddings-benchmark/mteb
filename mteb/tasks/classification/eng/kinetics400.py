from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@misc{kay2017kineticshumanactionvideo,
      title={The Kinetics Human Action Video Dataset},
      author={Will Kay and Joao Carreira and Karen Simonyan and Brian Zhang and Chloe Hillier and Sudheendra Vijayanarasimhan and Fabio Viola and Tim Green and Trevor Back and Paul Natsev and Mustafa Suleyman and Andrew Zisserman},
      year={2017},
      eprint={1705.06950},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1705.06950},
}
"""


class Kinetics400VAClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Kinetics400VA",
        description="Kinetics-400 is a large-scale action recognition dataset containing 400 human action classes from YouTube videos. Each clip is approximately 10 seconds long.",
        reference="https://arxiv.org/abs/1705.06950",
        dataset={
            "path": "mteb/kinetics-400",
            "revision": "e5b93b6eae80b8c9e9c88a381baae84d29b34fd2",
        },
        type="Classification",
        category="va2c",
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
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=CITATION,
    )

    input_column_name = ("video", "audio")
    label_column_name: str = "label"

    is_cross_validation: bool = False


class Kinetics400VClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Kinetics400V",
        description="Kinetics-400 is a large-scale action recognition dataset containing 400 human action classes from YouTube videos. Each clip is approximately 10 seconds long.",
        reference="https://arxiv.org/abs/1705.06950",
        dataset={
            "path": "mteb/kinetics-400",
            "revision": "e5b93b6eae80b8c9e9c88a381baae84d29b34fd2",
        },
        type="Classification",
        category="va2c",
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
        bibtex_citation=CITATION,
    )

    input_column_name = "video"
    label_column_name: str = "label"

    is_cross_validation: bool = False
