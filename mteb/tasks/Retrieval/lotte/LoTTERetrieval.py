from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)


class LoTTERetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        dataset={
            "url": "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz",
            "path": "colbertv2/lotte",
            "revision": "main",
        },
        name="LoTTE",
        description=(
            "Long-tail Topic-stratified Evaluation for IR featuring domain-specific datasets "
            "from StackExchange spanning writing, recreation, science, technology, and lifestyle. "
            "Includes both search-based queries from GooAQ and forum-based queries from StackExchange."
        ),
        type="Retrieval",
        modalities=["text"],
        category="s2s",
        reference="https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md",
        eval_splits=["dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="success@5",
        date=("2021-01-01", "2021-12-31"),
        domains=["Academic", "Web", "Social"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{santhanam2021colbertv2,
            title={ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction},
            author={Santhanam, Keshav and Khattab, Omar and Saad-Falcon, Jon and Potts, Christopher and Zaharia, Matei},
            journal={arXiv preprint arXiv:2112.01488},
            year={2021}
        }""",
        prompt=None,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_loaded = False

    def load_data(self, eval_splits: list | None = None, **kwargs) -> dict:
        if self.data_loaded:
            return {
                "queries": self.queries,
                "corpus": self.corpus,
                "relevant_docs": self.relevant_docs,
            }

        dataset_path = Path(self.metadata.dataset["path"])
        domains = ["writing", "recreation", "science", "technology", "lifestyle"]
        splits = eval_splits or self.metadata.eval_splits

        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}

        for split in splits:
            self.corpus[split] = {}
            self.queries[split] = {}
            self.relevant_docs[split] = {}

            for domain in domains:
                domain_path = dataset_path / domain / split
                corpus_file = domain_path / "collection.tsv"
                if corpus_file.exists():
                    with open(corpus_file, encoding="utf-8") as f:
                        self.corpus[split].update(
                            dict(line.strip().split("\t", 1) for line in f)
                        )

                search_queries_file = domain_path / "questions.search.tsv"
                if search_queries_file.exists():
                    with open(search_queries_file, encoding="utf-8") as f:
                        self.queries[split][f"{domain}.search"] = dict(
                            line.strip().split("\t", 1) for line in f
                        )

                forum_queries_file = domain_path / "questions.forum.tsv"
                if forum_queries_file.exists():
                    with open(forum_queries_file, encoding="utf-8") as f:
                        self.queries[split][f"{domain}.forum"] = dict(
                            line.strip().split("\t", 1) for line in f
                        )

                search_qas_file = domain_path / "qas.search.jsonl"
                if search_qas_file.exists():
                    with open(search_qas_file, encoding="utf-8") as f:
                        self.relevant_docs[split][f"{domain}.search"] = {
                            obj["qid"]: obj.get("answer_pids", [])
                            for obj in map(json.loads, f)
                        }

                forum_qas_file = domain_path / "qas.forum.jsonl"
                if forum_qas_file.exists():
                    with open(forum_qas_file, encoding="utf-8") as f:
                        self.relevant_docs[split][f"{domain}.forum"] = {
                            obj["qid"]: obj.get("answer_pids", [])
                            for obj in map(json.loads, f)
                        }

        self.data_loaded = True
        return {
            "queries": self.queries,
            "corpus": self.corpus,
            "relevant_docs": self.relevant_docs,
        }

    def evaluate(
        self,
        model,
        split: str = "test",
        encode_kwargs: dict | None = None,
        **kwargs,
    ) -> dict:
        encode_kwargs = encode_kwargs or {}

        data = self.load_data()
        # Merge queries and relevance judgments from nested dictionaries,
        # but leave corpus as-is since it is already a flat dictionary.
        dataset = {
            split: {
                "queries": {
                    k: v for d in data["queries"][split].values() for k, v in d.items()
                },
                "corpus": data["corpus"][split],
                "relevant": {
                    k: v
                    for d in data["relevant_docs"][split].values()
                    for k, v in d.items()
                },
            }
        }

        # Prepare lists for encoding while maintaining order.
        corpus_ids = list(dataset[split]["corpus"].keys())
        corpus_texts = list(dataset[split]["corpus"].values())
        query_ids = list(dataset[split]["queries"].keys())
        query_texts = list(dataset[split]["queries"].values())
        print(f"Loaded {len(corpus_texts)} passages and {len(query_texts)} queries.")

        corpus_embeddings = model.encode(
            corpus_texts, task_name="LoTTE", convert_to_numpy=True, **encode_kwargs
        )
        query_embeddings = model.encode(
            query_texts, task_name="LoTTE", convert_to_numpy=True, **encode_kwargs
        )

        if len(corpus_embeddings.shape) == 1:
            corpus_embeddings = corpus_embeddings.reshape(1, -1)
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        scores = np.matmul(query_embeddings, corpus_embeddings.T)

        k = 5
        success_at_k = 0
        for idx, pred_score in enumerate(scores):
            top_k_indices = np.argsort(-pred_score)[:k]
            top_k_pids = {corpus_ids[i] for i in top_k_indices}
            true_rel = set(dataset[split]["relevant"].get(query_ids[idx], []))
            if top_k_pids.intersection(true_rel):
                success_at_k += 1

        metric = success_at_k / len(query_ids) * 100
        print(f"Success@{k}: {metric:.2f}")

        return {split: {"default": {"success@5": metric}}}
