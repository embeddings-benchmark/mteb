from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GermanGovServiceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GermanGovServiceRetrieval",
        description="LHM-Dienstleistungen-QA is a German question answering dataset for government services of the Munich city administration. It associates questions with a textual context containing the answer",
        reference="https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA",
        dataset={
            "path": "mteb/GermanGovServiceRetrieval",
            "revision": "a6cf81304bf8b82d5497a0f7ad9e08399b8f27d3",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_5",
        date=("2022-11-01", "2022-11-30"),
        domains=["Government", "Written"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation=r"""
@software{lhm-dienstleistungen-qa,
  author = {Schröder, Leon Marius and
Gutknecht, Clemens and
Alkiddeh, Oubada and
Susanne Weiß,
Lukas, Leon},
  month = nov,
  publisher = {it@M},
  title = {LHM-Dienstleistungen-QA - german public domain question-answering dataset},
  url = {https://huggingface.co/datasets/it-at-m/LHM-Dienstleistungen-QA},
  year = {2022},
}
""",
        sample_creation="found",
    )
