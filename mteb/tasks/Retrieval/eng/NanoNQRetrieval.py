from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoNQRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoNQRetrieval",
        description="NanoNQ is a smaller subset of a dataset which contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question.",
        reference="https://ai.google.com/research/NaturalQuestions",
        dataset={
            "path": "zeta-alpha-ai/NanoNQ",
            "revision": "77540146379abf95df8326a3c5bb9eb21c7146c3",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2019-01-01", "2019-12-31"],
        domains=["Academic", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{47761,title	= {Natural Questions: a Benchmark for Question Answering Research},
        author	= {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
        and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
        and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
        and Slav Petrov},year	= {2019},journal	= {Transactions of the Association of Computational
        Linguistics}}""",
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
        adapted_from=["NQ"],
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoNQ",
            "corpus",
            revision="77540146379abf95df8326a3c5bb9eb21c7146c3",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoNQ",
            "queries",
            revision="77540146379abf95df8326a3c5bb9eb21c7146c3",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoNQ",
            "qrels",
            revision="77540146379abf95df8326a3c5bb9eb21c7146c3",
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
