from __future__ import annotations

import json
import logging
import os
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)

DOMAINS = ["writing", "recreation", "science", "technology", "lifestyle"]
DOMAINS_TYPES = ["search", "forum"]
HF_SUBSETS = [f"{d}_{t}" for d in DOMAINS for t in DOMAINS_TYPES]


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
        eval_splits=["test", "dev"],  # we assume evaluation is on the "test" split
        # For multilingual tasks, eval_langs is a dict mapping each domain to its language(s)
        eval_langs={domain: ["eng-Latn"] for domain in HF_SUBSETS},
        main_score="precision_at_5",
        date=("2021-12-02", "2022-06-10"),
        domains=["Academic", "Web", "Social"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{santhanam-etal-2022-colbertv2,
            title = "{C}ol{BERT}v2: Effective and Efficient Retrieval via Lightweight Late Interaction",
            author = "Santhanam, Keshav  and
              Khattab, Omar  and
              Saad-Falcon, Jon  and
              Potts, Christopher  and
              Zaharia, Matei",
            editor = "Carpuat, Marine  and
              de Marneffe, Marie-Catherine  and
              Meza Ruiz, Ivan Vladimir",
            booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
            month = jul,
            year = "2022",
            address = "Seattle, United States",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2022.naacl-main.272/",
            doi = "10.18653/v1/2022.naacl-main.272",
            pages = "3715--3734",
            abstract = "Neural information retrieval (IR) has greatly advanced search and other knowledge-intensive language tasks. While many neural IR methods encode queries and documents into single-vector representations, late interaction models produce multi-vector representations at the granularity of each token and decompose relevance modeling into scalable token-level computations. This decomposition has been shown to make late interaction more effective, but it inflates the space footprint of these models by an order of magnitude. In this work, we introduce ColBERTv2, a retriever that couples an aggressive residual compression mechanism with a denoised supervision strategy to simultaneously improve the quality and space footprint of late interaction. We evaluate ColBERTv2 across a wide range of benchmarks, establishing state-of-the-art quality within and outside the training domain while reducing the space footprint of late interaction models by 6{--}10x."
        }""",
        prompt=None,
    )

    def load_data(
        self, eval_splits: list | None = None, sample_limit: dict = None, **kwargs
    ):
        """Custom load_data that:
        - Downloads and extracts the dataset if not present.
        - Loads data only for the "test" split.
        - For each domain, loads:
            * Corpus from collection.tsv (or, if absent, from metadata.jsonl)
            * Search queries from questions.search.tsv
            * Forum queries from questions.forum.tsv
            * Qrels (relevant docs) from qas.search.jsonl and qas.forum.jsonl
        - Flattens the structure so that the final dictionaries are keyed by "domain_search" and "domain_forum".
        """
        if self.data_loaded:
            return

        dataset_info = self.metadata.dataset
        dataset_path = Path(dataset_info["path"])

        # Download and extract if necessary.
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

        # Temporary dictionaries to hold per-domain data.
        corpus_per_domain = defaultdict(dict)
        queries_per_domain = defaultdict(dict)
        relevant_docs_per_domain = defaultdict(dict)

        for domain in DOMAINS:
            for split in self.eval_splits:
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
                logger.info(
                    f"  Search queries file exists: {search_queries_file.exists()}"
                )
                logger.info(
                    f"  Forum queries file exists: {forum_queries_file.exists()}"
                )
                logger.info(f"  Search QAs file exists: {search_qas_file.exists()}")
                logger.info(f"  Forum QAs file exists: {forum_qas_file.exists()}")

                # Load corpus.
                corpus = {}
                if corpus_file.exists():
                    with open(corpus_file, encoding="utf-8") as f:
                        corpus = dict(
                            line.strip().split("\t", 1) for line in f if line.strip()
                        )
                elif metadata_file.exists():
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
                else:
                    logger.warning(f"No corpus file found for {domain} {split}.")
                corpus_per_domain[domain][split] = corpus

                # Load queries.
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
                queries_per_domain[domain][split] = queries

                # Load qrels.
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
                relevant_docs_per_domain[domain][split] = qrels

        # Flatten the dictionaries: create keys like "writing_search", "writing_forum", etc.
        final_corpus = {}
        final_queries = {}
        final_relevant_docs = {}
        for domain in DOMAINS:
            for split in self.eval_splits:
                current_corpus = corpus_per_domain[domain][split]
                current_queries = queries_per_domain[domain][split]
                current_relevant_docs = relevant_docs_per_domain[domain][split]
                # If corpus exists, replicate it for both search and forum if either queries exist.
                if "search" in current_queries:
                    final_corpus[f"{domain}_search"][split] = current_corpus
                if "forum" in current_queries:
                    final_corpus[f"{domain}_forum"][split] = current_corpus

                if "search" in current_queries:
                    final_queries[f"{domain}_search"] = current_queries["search"]
                if "forum" in current_queries:
                    final_queries[f"{domain}_forum"] = current_queries["forum"]

                if "search" in current_relevant_docs:
                    final_relevant_docs[f"{domain}_search"] = current_relevant_docs[
                        "search"
                    ]
                if "forum" in relevant_docs_per_domain[domain]:
                    final_relevant_docs[f"{domain}_forum"] = current_relevant_docs[
                        "forum"
                    ]

        self.data_loaded = True
        self.corpus = final_corpus
        self.queries = final_queries
        self.relevant_docs = final_relevant_docs
