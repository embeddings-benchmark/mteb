from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoFiQA2018Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoFiQA2018Retrieval",
        description="NanoFiQA2018 is a smaller subset of the Financial Opinion Mining and Question Answering dataset.",
        reference="https://sites.google.com/view/fiqa/",
        dataset={
            "path": "zeta-alpha-ai/NanoFiQA2018",
            "revision": "4163ba032953d5044a7a6244261413f609c14342",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2018-01-01", "2018-12-31"],
        domains=["Academic", "Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{
thakur2021beir,
title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
author={Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
year={2021},
url={https://openreview.net/forum?id=wCu6T5xFjeJ}
}""",
        prompt={
            "query": "Given a financial question, retrieve user replies that best answer the question"
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "zeta-alpha-ai/NanoFiQA2018",
            "corpus",
            revision="4163ba032953d5044a7a6244261413f609c14342",
        )
        self.queries = load_dataset(
            "zeta-alpha-ai/NanoFiQA2018",
            "queries",
            revision="4163ba032953d5044a7a6244261413f609c14342",
        )
        self.relevant_docs = load_dataset(
            "zeta-alpha-ai/NanoFiQA2018",
            "qrels",
            revision="4163ba032953d5044a7a6244261413f609c14342",
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

        self.relevant_docs = {
            split: {
                sample["query-id"]: {sample["corpus-id"]: 1}
                for sample in self.relevant_docs[split]
            }
            for split in self.relevant_docs
        }

        self.data_loaded = True
