from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SweFaqRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SweFaqRetrieval",
        dataset={
            "path": "mteb/SweFaqRetrieval",
            "revision": "208cb812631068a4bd8b93c8b9291370b47f282c",
        },
        description="A Swedish QA dataset derived from FAQ",
        reference="https://spraakbanken.gu.se/en/resources/superlim",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["swe-Latn"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2024-12-31"),  # best guess
        task_subtypes=["Question answering"],
        domains=["Government", "Non-fiction", "Written"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{berdivcevskis2023superlim,
  author = {Berdi{\v{c}}evskis, Aleksandrs and Bouma, Gerlof and Kurtz, Robin and Morger, Felix and {\"O}hman, Joey and Adesam, Yvonne and Borin, Lars and Dann{\'e}lls, Dana and Forsberg, Markus and Isbister, Tim and others},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages = {8137--8153},
  title = {Superlim: A Swedish language understanding evaluation benchmark},
  year = {2023},
}
""",  # for the benchmark in which this dataset is used
        prompt={"query": "Retrieve answers given questions in Swedish"},
    )
