from collections import defaultdict

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoQuoraRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoQuoraRetrieval",
        description="NanoQuoraRetrieval is a smaller subset of the "
        + "QuoraRetrieval dataset, which is based on questions that are marked as duplicates on the Quora platform. Given a"
        + " question, find other (duplicate) questions.",
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        dataset={
            "path": "zeta-alpha-ai/NanoQuoraRetrieval",
            "revision": "2ab2d73e6c862026282808b913a34f4136928545",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2017-01-01", "2017-12-31"],
        domains=["Social"],
        task_subtypes=["Duplicate Detection"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{quora-question-pairs,
  author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
  publisher = {Kaggle},
  title = {Quora Question Pairs},
  url = {https://kaggle.com/competitions/quora-question-pairs},
  year = {2017},
}
""",
        prompt={
            "query": "Given a question, retrieve questions that are semantically equivalent to the given question"
        },
        adapted_from=["QuoraRetrieval"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        corpus_hf = load_dataset(
            "zeta-alpha-ai/NanoQuoraRetrieval",
            "corpus",
            revision="2ab2d73e6c862026282808b913a34f4136928545",
        )
        queries_hf = load_dataset(
            "zeta-alpha-ai/NanoQuoraRetrieval",
            "queries",
            revision="2ab2d73e6c862026282808b913a34f4136928545",
        )
        qrels_hf = load_dataset(
            "zeta-alpha-ai/NanoQuoraRetrieval",
            "qrels",
            revision="2ab2d73e6c862026282808b913a34f4136928545",
        )

        self.dataset = {}
        for split in corpus_hf:
            corpus_ds = Dataset.from_list(
                [{"id": s["_id"], "text": s["text"]} for s in corpus_hf[split]]
            )
            queries_ds = Dataset.from_list(
                [{"id": s["_id"], "text": s["text"]} for s in queries_hf[split]]
            )
            relevant_docs: dict = defaultdict(dict)
            for query_id, corpus_id in zip(
                qrels_hf[split]["query-id"],
                qrels_hf[split]["corpus-id"],
            ):
                relevant_docs[query_id][corpus_id] = 1
            if "default" not in self.dataset:
                self.dataset["default"] = {}
            self.dataset["default"][split] = {
                "corpus": corpus_ds,
                "queries": queries_ds,
                "relevant_docs": dict(relevant_docs),
                "top_ranked": None,
            }

        self.data_loaded = True
