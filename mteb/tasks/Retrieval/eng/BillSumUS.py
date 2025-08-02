from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class BillSumUS(AbsTaskRetrieval):
    metadata = TaskMetadata(
        dataset={
            "path": "isaacus/mteb-BillSumUS",
            "revision": "0c063eb9b2f3085bbbc48f8d51f21a179254187e",
        },
        name="BillSumUS",
        description="A benchmark for retrieving US federal bills based on their summaries.",
        reference="https://huggingface.co/datasets/FiscalNote/billsum",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-08-14", "2025-07-18"),
        domains=["Legal", "Government"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Eidelman_2019,
  author = {Eidelman, Vladimir},
  booktitle = {Proceedings of the 2nd Workshop on New Frontiers in Summarization},
  doi = {10.18653/v1/d19-5406},
  pages = {48–56},
  publisher = {Association for Computational Linguistics},
  title = {BillSum: A Corpus for Automatic Summarization of US Legislation},
  url = {http://dx.doi.org/10.18653/v1/D19-5406},
  year = {2019},
}
""",
    )
