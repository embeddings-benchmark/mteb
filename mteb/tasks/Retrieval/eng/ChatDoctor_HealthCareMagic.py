from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class ChatDoctor_HealthCareMagic(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChatDoctor_HealthCareMagic",
        description="The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.",
        reference="https://github.com/Kent0n-Li/ChatDoctor",
        dataset={
            "path": "embedding-benchmark/ChatDoctor_HealthCareMagic",
            "revision": "main",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-05-20", "2021-05-20"),
        domains=["Medical"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{hendrycksapps2021,
  author = {Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal = {NeurIPS},
  title = {Measuring Coding Challenge Competence With APPS},
  year = {2021},
}
""",
    )
