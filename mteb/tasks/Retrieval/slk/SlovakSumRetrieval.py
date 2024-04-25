from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SlovakSumRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SlovakSumRetrieval",
        description="""
            SlovakSum, a Slovak news summarization dataset consisting of over 200 thousand
            news articles with titles and short abstracts obtained from multiple Slovak newspapers.

            Originally intended as a summarization task, but since no human annotations were provided
            here reformulated to a retrieval task.
        """,
        reference="https://huggingface.co/datasets/kiviki/SlovakSum",
        dataset={
            "path": "kiviki/SlovakSum",
            "revision": "85d6b32f2762313714618171b9d1a65eb7408835",
        },
        type="Retrieval",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="ndcg_at_1",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="openrail",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation="""
            @inproceedings{OndrejowaSlovakSum24,
                title = {SlovakSum: A Large Scale Slovak Summarization Dataset},
                booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation},
                author = {Ondrejová, Viktória and Šuppa, Marek},
                date = {2024},
            }
        """,
        n_samples={"test": 600},
        avg_character_length={"test": 238.44},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        n_sample = self.metadata_dict["n_samples"]["test"]

        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            split_ds = datasets.load_dataset(
                dataset_path, split=f"{split}[:{n_sample}]"
            )
            # Transforming news summary into retrieval task
            queries = {f"q{e+1}": x["sum"] for e, x in enumerate(split_ds)}
            corpus = {
                f"d{e+1}": {"title": x["title"], "text": x["text"]}
                for e, x in enumerate(split_ds)
            }
            qrels = {f"q{i+1}": {f"d{i+1}": 1} for i in range(split_ds.shape[0])}
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )
        self.data_loaded = True
