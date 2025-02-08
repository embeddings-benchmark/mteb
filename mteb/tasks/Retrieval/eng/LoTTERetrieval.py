from __future__ import annotations

import json
import logging
import os
import tarfile
import urllib.request
from pathlib import Path

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)


class LoTTERetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LoTTE",
        dataset={
            "url": "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz",
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
        date=("2021-12-02", "2022-06-10"),
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
        """Custom load_data that downloads the dataset from the provided URL if the dataset folder
        does not exist, then reads the files for each domain and split.
        """
        if self.data_loaded:
            return {
                "queries": self.queries,
                "corpus": self.corpus,
                "relevant_docs": self.relevant_docs,
            }

        dataset_info = self.metadata.dataset
        dataset_path = Path(dataset_info["path"])

        if not dataset_path.exists():
            url = dataset_info.get("url")
            if not url:
                raise ValueError(
                    "No URL provided in metadata.dataset to download the dataset."
                )
            logger.info(
                f"Dataset path {dataset_path} not found. Downloading from {url}..."
            )
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            tar_file = dataset_path.parent / "lotte.tar.gz"
            urllib.request.urlretrieve(url, tar_file)
            logger.info("Extracting dataset...")
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(path=dataset_path.parent)
            os.remove(tar_file)
            logger.info("Dataset downloaded and extracted.")

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
                search_queries_file = domain_path / "questions.search.tsv"
                forum_queries_file = domain_path / "questions.forum.tsv"
                search_qas_file = domain_path / "qas.search.jsonl"
                forum_qas_file = domain_path / "qas.forum.jsonl"

                logger.info(f"Checking files in {domain_path}:")
                logger.info(f"  Corpus file exists: {corpus_file.exists()}")
                logger.info(
                    f"  Search queries file exists: {search_queries_file.exists()}"
                )
                logger.info(
                    f"  Forum queries file exists: {forum_queries_file.exists()}"
                )
                logger.info(f"  Search QAs file exists: {search_qas_file.exists()}")
                logger.info(f"  Forum QAs file exists: {forum_qas_file.exists()}")

                if corpus_file.exists():
                    with open(corpus_file, encoding="utf-8") as f:
                        for line in f:
                            doc_id, text = line.strip().split("\t", 1)
                            self.corpus[split][doc_id] = text

                if search_queries_file.exists():
                    with open(search_queries_file, encoding="utf-8") as f:
                        self.queries[split][f"{domain}.search"] = dict(
                            line.strip().split("\t", 1) for line in f
                        )

                if forum_queries_file.exists():
                    with open(forum_queries_file, encoding="utf-8") as f:
                        self.queries[split][f"{domain}.forum"] = dict(
                            line.strip().split("\t", 1) for line in f
                        )

                if search_qas_file.exists():
                    with open(search_qas_file, encoding="utf-8") as f:
                        self.relevant_docs[split][f"{domain}.search"] = {
                            obj["qid"]: obj.get("answer_pids", [])
                            for obj in map(json.loads, f)
                        }

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
