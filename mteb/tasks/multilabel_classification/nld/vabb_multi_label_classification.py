from mteb.abstasks.multilabel_classification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class VABBMultiLabelClassification(AbsTaskMultilabelClassification):
    samples_per_label = 128
    metadata = TaskMetadata(
        name="VABBMultiLabelClassification",
        dataset={
            "path": "clips/mteb-nl-vabb-mlcls-pr",
            "revision": "584c70f5104671772119f21e9f8a3c912ac07d4a",
        },
        description="This dataset contains the fourteenth edition of the Flemish Academic Bibliography for the Social "
        "Sciences and Humanities (VABB-SHW), a database of academic publications from the social sciences "
        "and humanities authored by researchers affiliated to Flemish universities (more information). "
        "Publications in the database are used as one of the parameters of the Flemish performance-based "
        "research funding system",
        reference="https://zenodo.org/records/14214806",
        type="MultilabelClassification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        date=("2020-01-01", "2021-04-01"),
        domains=["Academic", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@dataset{aspeslagh2024vabb,
  author = {Aspeslagh, Pieter and Guns, Raf and Engels, Tim C. E.},
  doi = {10.5281/zenodo.14214806},
  publisher = {Zenodo},
  title = {VABB-SHW: Dataset of Flemish Academic Bibliography for the Social Sciences and Humanities (edition 14)},
  url = {https://doi.org/10.5281/zenodo.14214806},
  year = {2024},
}
""",
        prompt={
            "query": "Classificeer de onderwerpen van een wetenschappelijk artikel op basis van de abstract"
        },
    )
