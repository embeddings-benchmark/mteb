from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SweRecClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SweRecClassification",
        description="A Swedish dataset for sentiment classification on review",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        dataset={
            "path": "mteb/swerec_classification",
            "revision": "b07c6ce548f6a7ac8d546e1bbe197a0086409190",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["swe-Latn"],
        main_score="accuracy",
        date=("2023-01-01", "2023-12-31"),  # based on the publication date
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{nielsen-2023-scandeval,
  address = {T{\'o}rshavn, Faroe Islands},
  author = {Nielsen, Dan},
  booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
  editor = {Alum{\"a}e, Tanel  and
Fishel, Mark},
  month = may,
  pages = {185--201},
  publisher = {University of Tartu Library},
  title = {{S}cand{E}val: A Benchmark for {S}candinavian Natural Language Processing},
  url = {https://aclanthology.org/2023.nodalida-1.20},
  year = {2023},
}
""",
        prompt="Classify Swedish reviews by sentiment",
    )
