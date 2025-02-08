from __future__ import annotations

import json
import logging
from pathlib import Path

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)


class LoTTERetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LoTTE",
        dataset={
            "path": "colbertv2/lotte",
            "revision": "main",
        },
        description=(
            "LoTTE (Long-Tail Topic-stratified Evaluation for IR) is designed to evaluate retrieval models "
            "on underrepresented, long-tail topics. Unlike MSMARCO or BEIR, LoTTE features domain-specific queries and "
            "passages from StackExchange (covering writing, recreation, science, technology, and lifestyle), providing "
            "a challenging out-of-domain generalization benchmark."
        ),
        type="Retrieval",
        modalities=["text"],
        category="s2s",
        reference="https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        eval_langs_per_domain={
            "writing": ["eng-Latn"],
            "recreation": ["eng-Latn"],
            "science": ["eng-Latn"],
            "technology": ["eng-Latn"],
            "lifestyle": ["eng-Latn"],
        },
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

    def load_data(self, eval_splits: list | None = None, **kwargs) -> dict:
        """Override load_data to ensure correct corpus, queries, and qrels are loaded without conflicts."""
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
                        for line in f:
                            doc_id, text = line.strip().split("\t", 1)
                            self.corpus[split][doc_id] = text

                search_queries_file = domain_path / "questions.search.tsv"
                if search_queries_file.exists():
                    with open(search_queries_file, encoding="utf-8") as f:
                        self.queries[split][f"{domain}.search"] = self.queries[split][
                            f"{domain}.search"
                        ] = dict(line.strip().split("\t", 1) for line in f)

                forum_queries_file = domain_path / "questions.forum.tsv"
                if forum_queries_file.exists():
                    with open(forum_queries_file, encoding="utf-8") as f:
                        self.queries[split][f"{domain}.forum"] = self.queries[split][
                            f"{domain}.forum"
                        ] = dict(line.strip().split("\t", 1) for line in f)

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

    def dataset_transform(self, data: dict) -> dict:
        """Transform dataset to merge nested dictionaries for queries and relevant docs while leaving corpus unchanged."""
        split = self.metadata.eval_splits[0]
        return {
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
