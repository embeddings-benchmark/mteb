import logging

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)


class VoxPopuliAccentPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="VoxPopuliAccentPairClassification",
        description="Classifying same or different regional accent of English",
        reference="https://aclanthology.org/2021.acl-long.80/",
        dataset={
            "path": "mteb/VoxPopuliAccentPairClassification",
            "revision": "cc395e3ab521e1bcd4ddbb05f2810acf807297ec",
        },
        type="AudioPairClassification",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2021-01-01", "2021-08-01"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{wang-etal-2021-voxpopuli,
  address = {Online},
  author = {Wang, Changhan  and
Riviere, Morgane  and
Lee, Ann  and
Wu, Anne  and
Talnikar, Chaitanya  and
Haziza, Daniel  and
Williamson, Mary  and
Pino, Juan  and
Dupoux, Emmanuel},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  doi = {10.18653/v1/2021.acl-long.80},
  editor = {Zong, Chengqing  and
Xia, Fei  and
Li, Wenjie  and
Navigli, Roberto},
  month = aug,
  pages = {993--1003},
  publisher = {Association for Computational Linguistics},
  title = {{V}ox{P}opuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation},
  url = {https://aclanthology.org/2021.acl-long.80/},
  year = {2021},
}
""",
    )

    input1_column_name: str = "audio1"
    input2_column_name: str = "audio2"
    label_column_name: str = "label"
