from __future__ import annotations

import json
import logging
import os
import tarfile
import urllib.request
from pathlib import Path

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)


class LoTTERetrieval(MultilingualTask, AbsTaskRetrieval):
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
        eval_splits=["test"],  # we only load "test" for evaluation
        # For multilingual tasks, eval_langs is a dict mapping each domain to its language list.
        eval_langs={
            "writing": ["eng-Latn"],
            "recreation": ["eng-Latn"],
            "science": ["eng-Latn"],
            "technology": ["eng-Latn"],
            "lifestyle": ["eng-Latn"],
        },
        main_score="precision_at_5",
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

    # Override hf_subsets property to return domain names.
    @property
    def hf_subsets(self) -> list[str]:
        return list(self.metadata.eval_langs.keys())

    @hf_subsets.setter
    def hf_subsets(self, value):
        pass

    def load_data(self, eval_splits: list | None = None, **kwargs) -> dict:
        """Custom load_data that:
        - Downloads and extracts the dataset if not present.
        - Iterates over each domain (e.g., "writing", "recreation", etc.) in the "test" split.
        - For each domain, loads:
            * Corpus from collection.tsv if available, else metadata.jsonl.
            * Queries: loads search queries and forum queries into separate dictionaries.
            * Qrels (relevant docs): loads search QAs and forum QAs into separate dictionaries,
              converting answer lists into dictionaries (with value 1).
        - Stores the data per domain.
        """
        # We assume evaluation is on the "test" split.
        split = "test"
        if self.data_loaded:
            return {
                "queries": self.queries,
                "corpus": self.corpus,
                "relevant_docs": self.relevant_docs,
            }
        dataset_info = self.metadata.dataset
        dataset_path = Path(dataset_info["path"])

        # Auto-download and extract if dataset is not present.
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

        # Define expected domains.
        domains = ["writing", "recreation", "science", "technology", "lifestyle"]

        # Initialize dictionaries keyed by domain.
        self.corpus = {domain: {} for domain in domains}
        self.queries = {domain: {} for domain in domains}
        self.relevant_docs = {domain: {} for domain in domains}

        for domain in domains:
            domain_path = dataset_path / domain / split
            corpus_file = domain_path / "collection.tsv"
            metadata_file = domain_path / "metadata.jsonl"
            search_queries_file = domain_path / "questions.search.tsv"
            forum_queries_file = domain_path / "questions.forum.tsv"
            search_qas_file = domain_path / "qas.search.jsonl"
            forum_qas_file = domain_path / "qas.forum.jsonl"

            logger.info(f"Checking files in {domain_path}:")
            logger.info(f"  Corpus file exists: {corpus_file.exists()}")
            logger.info(f"  Metadata file exists: {metadata_file.exists()}")
            logger.info(f"  Search queries file exists: {search_queries_file.exists()}")
            logger.info(f"  Forum queries file exists: {forum_queries_file.exists()}")
            logger.info(f"  Search QAs file exists: {search_qas_file.exists()}")
            logger.info(f"  Forum QAs file exists: {forum_qas_file.exists()}")

            # Load corpus: prefer collection.tsv; otherwise, use metadata.jsonl.
            if corpus_file.exists():
                with open(corpus_file, encoding="utf-8") as f:
                    self.corpus[domain] = dict(
                        line.strip().split("\t", 1) for line in f if line.strip()
                    )
            elif metadata_file.exists():
                corpus = {}
                with open(metadata_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            doc_id = obj.get("pid") or obj.get("id")
                            text = obj.get("text") or obj.get("body")
                            if doc_id and text:
                                corpus[doc_id] = text
                        except Exception as e:
                            logger.error(f"Error parsing {metadata_file}: {e}")
                self.corpus[domain] = corpus
            else:
                logger.warning(f"No corpus file found for {domain} {split}.")

            # Load queries: merge search and forum queries into two separate keys.
            queries = {}
            if search_queries_file.exists():
                with open(search_queries_file, encoding="utf-8") as f:
                    queries["search"] = dict(
                        line.strip().split("\t", 1) for line in f if line.strip()
                    )
            if forum_queries_file.exists():
                with open(forum_queries_file, encoding="utf-8") as f:
                    queries["forum"] = dict(
                        line.strip().split("\t", 1) for line in f if line.strip()
                    )
            self.queries[domain] = queries

            # Load qrels (relevant docs): merge search and forum QAs into separate keys.
            qrels = {}
            if search_qas_file.exists():
                with open(search_qas_file, encoding="utf-8") as f:
                    qrels["search"] = {
                        str(obj["qid"]): {
                            str(pid): 1 for pid in obj.get("answer_pids", [])
                        }
                        for obj in map(json.loads, f)
                    }
            if forum_qas_file.exists():
                with open(forum_qas_file, encoding="utf-8") as f:
                    qrels["forum"] = {
                        str(obj["qid"]): {
                            str(pid): 1 for pid in obj.get("answer_pids", [])
                        }
                        for obj in map(json.loads, f)
                    }
            self.relevant_docs[domain] = qrels

        self.data_loaded = True
        return {
            "queries": self.queries,
            "corpus": self.corpus,
            "relevant_docs": self.relevant_docs,
        }

    def dataset_transform(self, data: dict) -> dict:
        """For evaluation, return the per-domain structure without merging.
        Each domain is a separate hf_subset.
        """
        return data
