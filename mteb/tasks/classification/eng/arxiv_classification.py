from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2c",
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
        superseded_by="ArxivClassification.v2",
    )


class ArxivClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ArxivClassification.v2",
        description="Classification Dataset of Arxiv Papers This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        dataset={
            "path": "mteb/arxiv",
            "revision": "202e10e9a5d37a5068397b48184d0728346a7b4a",
        },
        reference="https://ieeexplore.ieee.org/document/8675939",
        type="Classification",
        category="t2c",
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
        adapted_from=["ArxivClassification"],
    )
