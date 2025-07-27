from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ArxivClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ArxivClassification",
        description="Classification Dataset of Arxiv Papers",
        dataset={
            "path": "mteb/ArxivClassification",
            "revision": "5e80893bf045abefbf8cbe5d713bddc91ae158d5",
        },
        reference="https://ieeexplore.ieee.org/document/8675939",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1998-11-11", "2019-03-28"),
        domains=["Academic", "Written"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{8675939,
  author = {He, Jun and Wang, Liqun and Liu, Liu and Feng, Jiao and Wu, Hao},
  doi = {10.1109/ACCESS.2019.2907992},
  journal = {IEEE Access},
  number = {},
  pages = {40707-40718},
  title = {Long Document Classification From Local Word Glimpses via Recurrent Attention Learning},
  volume = {7},
  year = {2019},
}
""",
    )
