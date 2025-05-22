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
        reference="https://huggingface.co/datasets/NaiveNeuron/slovaksum",
        dataset={
            "path": "NaiveNeuron/slovaksum",
            "revision": "85d6b32f2762313714618171b9d1a65eb7408835",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="ndcg_at_10",
        date=("2015-04-26", "2022-01-11"),
        domains=["News", "Social", "Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="openrail",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{OndrejowaSlovakSum24,
  author = {Ondrejová, Viktória and Šuppa, Marek},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation},
  date = {2024},
  title = {SlovakSum: A Large Scale Slovak Summarization Dataset},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        n_sample = 600

        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            split_ds = datasets.load_dataset(
                dataset_path, split=f"{split}[:{n_sample}]"
            )
            # Transforming news summary into retrieval task
            queries = {f"q{e + 1}": x["sum"] for e, x in enumerate(split_ds)}
            corpus = {
                f"d{e + 1}": {"title": x["title"], "text": x["text"]}
                for e, x in enumerate(split_ds)
            }
            qrels = {f"q{i + 1}": {f"d{i + 1}": 1} for i in range(split_ds.shape[0])}
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )
        self.data_loaded = True
