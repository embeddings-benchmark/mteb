from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


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
        category="s2s",
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
        bibtex_citation="""@misc{quora-question-pairs,
    author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
    title = {Quora Question Pairs},
    publisher = {Kaggle},
    year = {2017},
    url = {https://kaggle.com/competitions/quora-question-pairs}
}""",
        prompt={
            "query": "Given a question, retrieve questions that are semantically equivalent to the given question"
        },
        adapted_from=["QuoraRetrieval"],
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoQuoraRetrieval",
            "corpus",
            revision="2ab2d73e6c862026282808b913a34f4136928545",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoQuoraRetrieval",
            "queries",
            revision="2ab2d73e6c862026282808b913a34f4136928545",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoQuoraRetrieval",
            "qrels",
            revision="2ab2d73e6c862026282808b913a34f4136928545",
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
