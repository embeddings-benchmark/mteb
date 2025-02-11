from __future__ import annotations
from pathlib import Path
import os
import urllib.request
import tarfile
import json
import logging

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
        eval_splits=["test"],
        # For multilingual tasks, eval_langs is a dict mapping each domain to its languages.
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

    def load_data(self, eval_splits: list | None = None, **kwargs) -> dict:
        """
        Custom load_data that:
          - Auto-downloads and extracts the dataset if not present.
          - Iterates over domains first and then over splits.
          - Loads corpus (collection.tsv), queries (questions.search.tsv and questions.forum.tsv),
            and qrels (qas.search.jsonl and qas.forum.jsonl) for each domain and split.
          - Converts the qrels answer_pids from a list to a dict mapping each doc_id to a relevance score (1).
          - Merges the per-domain data into dictionaries keyed by domain.
        """
        if self.data_loaded:
            return {
                "queries": self.queries,
                "corpus": self.corpus,
                "relevant_docs": self.relevant_docs,
            }

        dataset_info = self.metadata.dataset
        dataset_path = Path(dataset_info["path"])

        # Auto-download and extract dataset if not present.
        if not dataset_path.exists():
            url = dataset_info.get("url")
            if not url:
                raise ValueError("No URL provided in metadata.dataset to download the dataset.")
            logger.info(f"Dataset path {dataset_path} not found. Downloading from {url}...")
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

        # Initialize dictionaries keyed by domain.
        self.corpus = {domain: {split: {} for split in splits} for domain in domains}
        self.queries = {domain: {split: {} for split in splits} for domain in domains}
        self.relevant_docs = {domain: {split: {} for split in splits} for domain in domains}

        for domain in domains:
            for split in splits:
                domain_path = dataset_path / domain / split
                corpus_file = domain_path / "collection.tsv"
                search_queries_file = domain_path / "questions.search.tsv"
                forum_queries_file = domain_path / "questions.forum.tsv"
                search_qas_file = domain_path / "qas.search.jsonl"
                forum_qas_file = domain_path / "qas.forum.jsonl"

                logger.info(f"Checking files in {domain_path}:")
                logger.info(f"  Corpus file exists: {corpus_file.exists()}")
                logger.info(f"  Search queries file exists: {search_queries_file.exists()}")
                logger.info(f"  Forum queries file exists: {forum_queries_file.exists()}")
                logger.info(f"  Search QAs file exists: {search_qas_file.exists()}")
                logger.info(f"  Forum QAs file exists: {forum_qas_file.exists()}")

                if corpus_file.exists():
                    with open(corpus_file, encoding="utf-8") as f:
                        # Each line: doc_id<TAB>text
                        self.corpus[domain][split] = dict(
                            line.strip().split("\t", 1) for line in f
                        )

                if search_queries_file.exists():
                    with open(search_queries_file, encoding="utf-8") as f:
                        self.queries[domain][split] = dict(
                            line.strip().split("\t", 1) for line in f
                        )

                if forum_queries_file.exists():
                    with open(forum_queries_file, encoding="utf-8") as f:
                        # Store forum queries under a distinct key.
                        self.queries[domain][f"{split}.forum"] = dict(
                            line.strip().split("\t", 1) for line in f
                        )

                if search_qas_file.exists():
                    with open(search_qas_file, encoding="utf-8") as f:
                        # Convert the answer_pids list into a dictionary: {doc_id: 1}
                        self.relevant_docs[domain][split] = {
                            str(obj["qid"]): {str(pid): 1 for pid in obj.get("answer_pids", [])}
                            for obj in map(json.loads, f)
                        }

                if forum_qas_file.exists():
                    with open(forum_qas_file, encoding="utf-8") as f:
                        self.relevant_docs[domain][f"{split}.forum"] = {
                            str(obj["qid"]): {str(pid): 1 for pid in obj.get("answer_pids", [])}
                            for obj in map(json.loads, f)
                        }

        self.data_loaded = True
        return {
            "queries": self.queries,
            "corpus": self.corpus,
            "relevant_docs": self.relevant_docs,
        }

    def dataset_transform(self, data: dict) -> dict:
        """
        Merge the per-domain data for the chosen split across all domains.
        This produces a merged view with keys "queries", "corpus", and "relevant".
        """
        split = self.metadata.eval_splits[0]
        merged_queries = {}
        merged_corpus = {}
        merged_relevant = {}
        for domain in self.queries:
            if split in self.queries[domain]:
                merged_queries.update(self.queries[domain][split])
            for key, value in self.queries[domain].items():
                if key.startswith(split) and key != split:
                    merged_queries.update(value)
        for domain in self.corpus:
            if split in self.corpus[domain]:
                merged_corpus.update(self.corpus[domain][split])
        for domain in self.relevant_docs:
            if split in self.relevant_docs[domain]:
                merged_relevant.update(self.relevant_docs[domain][split])
            for key, value in self.relevant_docs[domain].items():
                if key.startswith(split) and key != split:
                    merged_relevant.update(value)
        return {
            split: {
                "queries": merged_queries,
                "corpus": merged_corpus,
                "relevant": merged_relevant,
            }
        }