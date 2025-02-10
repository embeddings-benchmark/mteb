from __future__ import annotations
from pathlib import Path
import os
import json
import logging

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
        """
        Custom load_data that loads the dataset from local files.
        If the dataset folder does not exist, you can optionally add auto-download logic.
        This implementation iterates over domains first, then splits, and finally merges data
        into a structure keyed by split.
        """
        if self.data_loaded:
            return {
                "queries": self.queries,
                "corpus": self.corpus,
                "relevant_docs": self.relevant_docs,
            }
        
        dataset_path = Path(self.metadata.dataset["path"])
        # (Optional) Auto-download if the dataset folder is missing:
        if not dataset_path.exists():
            url = self.metadata.dataset.get("url")
            if not url:
                raise ValueError("No URL provided in metadata.dataset to download the dataset.")
            logger.info(f"Dataset path {dataset_path} not found. Downloading from {url}...")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            tar_file = dataset_path.parent / "lotte.tar.gz"
            import urllib.request, tarfile
            urllib.request.urlretrieve(url, tar_file)
            logger.info("Extracting dataset...")
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(path=dataset_path.parent)
            os.remove(tar_file)
            logger.info("Dataset downloaded and extracted.")
        
        domains = ["writing", "recreation", "science", "technology", "lifestyle"]
        splits = eval_splits or self.metadata.eval_splits

        # Initialize per-domain dictionaries
        domain_corpus = {}
        domain_queries = {}
        domain_relevant = {}
        for domain in domains:
            domain_corpus[domain] = {}
            domain_queries[domain] = {}
            domain_relevant[domain] = {}
            for split in splits:
                domain_path = dataset_path / domain / split
                # Build keys for files
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
                        # Expecting each line as: doc_id<TAB>text
                        domain_corpus[domain][split] = dict(
                            line.strip().split("\t", 1) for line in f
                        )

                if search_queries_file.exists():
                    with open(search_queries_file, encoding="utf-8") as f:
                        domain_queries[domain][split] = dict(
                            line.strip().split("\t", 1) for line in f
                        )

                if forum_queries_file.exists():
                    with open(forum_queries_file, encoding="utf-8") as f:
                        # Use a different key for forum queries; we append ".forum" to distinguish
                        domain_queries[domain][f"{split}.forum"] = dict(
                            line.strip().split("\t", 1) for line in f
                        )

                if search_qas_file.exists():
                    with open(search_qas_file, encoding="utf-8") as f:
                        domain_relevant[domain][split] = {
                            obj["qid"]: obj.get("answer_pids", [])
                            for obj in map(json.loads, f)
                        }

                if forum_qas_file.exists():
                    with open(forum_qas_file, encoding="utf-8") as f:
                        domain_relevant[domain][f"{split}.forum"] = {
                            obj["qid"]: obj.get("answer_pids", [])
                            for obj in map(json.loads, f)
                        }
        
        # Merge per-domain data into a single dictionary keyed by split
        merged_data = {"queries": {}, "corpus": {}, "relevant_docs": {}}
        for split in splits:
            merged_data["corpus"][split] = {}
            merged_data["queries"][split] = {}
            merged_data["relevant_docs"][split] = {}
            for domain in domains:
                # Merge corpus if present
                if split in domain_corpus[domain]:
                    merged_data["corpus"][split].update(domain_corpus[domain][split])
                # Merge search queries if present
                if split in domain_queries[domain]:
                    merged_data["queries"][split].update(domain_queries[domain][split])
                # Merge forum queries if present (keys like "test.forum")
                for key, value in domain_queries[domain].items():
                    if key.startswith(split) and key != split:
                        merged_data["queries"][split].update(value)
                # Merge relevant docs (search and forum)
                if split in domain_relevant[domain]:
                    merged_data["relevant_docs"][split].update(domain_relevant[domain][split])
                for key, value in domain_relevant[domain].items():
                    if key.startswith(split) and key != split:
                        merged_data["relevant_docs"][split].update(value)
        
        self.data_loaded = True
        self.queries = merged_data["queries"]
        self.corpus = merged_data["corpus"]
        self.relevant_docs = merged_data["relevant_docs"]
        return merged_data

    def dataset_transform(self, data: dict) -> dict:
        split = self.metadata.eval_splits[0]
        return {
            split: {
                "queries": {
                    k: v for d in data["queries"][split].values() for k, v in d.items()
                },
                "corpus": data["corpus"][split],
                "relevant": {
                    k: v for d in data["relevant_docs"][split].values() for k, v in d.items()
                },
            }
        }
