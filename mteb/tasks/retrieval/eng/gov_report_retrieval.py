from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GovReportRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        dataset={
            "path": "isaacus/mteb-GovReport",
            "revision": "4482db2a053706c0aef0ab7bf1878d29bb0295f8",
        },
        name="GovReport",
        description="A dataset for evaluating the ability of information retrieval models to retrieve lengthy US government reports from their summaries.",
        reference="https://huggingface.co/datasets/launch/gov_report",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-06-06", "2025-07-28"),
        domains=["Legal", "Government"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{huang-etal-2021-efficient,
  address = {Online},
  author = {Huang, Luyang  and
Cao, Shuyang  and
Parulian, Nikolaus  and
Ji, Heng  and
Wang, Lu},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  doi = {10.18653/v1/2021.naacl-main.112},
  eprint = {2104.02112},
  month = jun,
  pages = {1419--1436},
  publisher = {Association for Computational Linguistics},
  title = {Efficient Attentions for Long Document Summarization},
  url = {https://aclanthology.org/2021.naacl-main.112},
  year = {2021},
}
""",
    )
