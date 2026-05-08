from collections import defaultdict

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoArguAnaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoArguAnaRetrieval",
        description="NanoArguAna is a smaller subset of ArguAna, a dataset for argument retrieval in debate contexts.",
        reference="http://argumentation.bplaced.net/arguana/data",
        dataset={
            "path": "zeta-alpha-ai/NanoArguAna",
            "revision": "8f4a982d470a32c45817738b9d29042ca55d75ad",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2020-01-01", "2020-12-31"],
        domains=["Social", "Web", "Written"],
        task_subtypes=["Discourse coherence"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{wachsmuth2018retrieval,
  author = {Wachsmuth, Henning and Syed, Shahbaz and Stein, Benno},
  booktitle = {ACL},
  title = {Retrieval of the Best Counterargument without Prior Topic Knowledge},
  year = {2018},
}
""",
        prompt={"query": "Given a claim, find documents that refute the claim"},
        adapted_from=["ArguAna"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        corpus_hf = load_dataset(
            "zeta-alpha-ai/NanoArguAna",
            "corpus",
            revision="8f4a982d470a32c45817738b9d29042ca55d75ad",
        )
        queries_hf = load_dataset(
            "zeta-alpha-ai/NanoArguAna",
            "queries",
            revision="8f4a982d470a32c45817738b9d29042ca55d75ad",
        )
        qrels_hf = load_dataset(
            "zeta-alpha-ai/NanoArguAna",
            "qrels",
            revision="8f4a982d470a32c45817738b9d29042ca55d75ad",
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
