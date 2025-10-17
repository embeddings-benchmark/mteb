from collections import defaultdict

from datasets import load_dataset

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

    def load_data(self) -> None:
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoArguAna",
            "corpus",
            revision="8f4a982d470a32c45817738b9d29042ca55d75ad",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoArguAna",
            "queries",
            revision="8f4a982d470a32c45817738b9d29042ca55d75ad",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoArguAna",
            "qrels",
            revision="8f4a982d470a32c45817738b9d29042ca55d75ad",
        )

        self.corpus = {
            split: {
                sample["_id"]: {"_id": sample["_id"], "text": sample["text"]}
                for sample in self.corpus[split]
            }
            for split in self.corpus
        }

        self.queries = {
            split: {sample["_id"]: sample["text"] for sample in self.queries[split]}
            for split in self.queries
        }

        relevant_docs = {}

        for split in self.relevant_docs:
            relevant_docs[split] = defaultdict(dict)
            for query_id, corpus_id in zip(
                self.relevant_docs[split]["query-id"],
                self.relevant_docs[split]["corpus-id"],
            ):
                relevant_docs[split][query_id][corpus_id] = 1
        self.relevant_docs = relevant_docs

        self.data_loaded = True
