from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

DOMAINS = [
    "fiqa",
    "nfcorpus",
    "scifact_open",
    "aila",
    "fire",
    "pm",
    "cds"
]

DOMAINS_langs = {split: ["eng"] for split in DOMAINS}

class IFRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IFRetrieval",
        dataset={
            "path": "if-ir/ifir",
            "revision": "1b0c836",
        },
        reference="https://huggingface.co/datasets/if-ir/ifir",
        description="IFIR retrieval dataset.",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=DOMAINS_langs,
        main_score="ndcg_at_20",
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text"],
        bibtex_citation=r"""@inproceedings{song2025ifir,
  title={IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
  author={Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={10186--10204},
  year={2025}
}""",
    )
    
    def load_ifir_data(
        self,
        path: str,
        domains: list,
        eval_splits: list,
        cache_dir: str | None = None,
        revision: str | None = None,
    ):
        corpus = {domain: {split: None for split in eval_splits} for domain in DOMAINS}
        queries = {domain: {split: None for split in eval_splits} for domain in DOMAINS}
        relevant_docs = {
            domain: {split: None for split in eval_splits} for domain in DOMAINS
        }

        for domain in domains:
            domain_corpus = datasets.load_dataset(
                path, "corpus", split=domain, cache_dir=cache_dir, revision=revision
            )
            domain_queries = datasets.load_dataset(
                path, "queries", split=domain, cache_dir=cache_dir, revision=revision
            )
            qrels = datasets.load_dataset(
                path, "qrels", split=domain, cache_dir=cache_dir, revision=revision
            )
            corpus[domain]["test"] = {
                e["_id"]: {"text": e["text"]} for e in domain_corpus
            }
            queries[domain]["test"] = {
                e["_id"]: e["text"] for e in domain_queries
            }
            relevant_docs[domain]["test"] = {}

            for e in qrels:
                qid = e["query-id"]
                doc_id = e["corpus-id"]
                if qid not in relevant_docs[domain]["test"]:
                    relevant_docs[domain]["test"][qid] = defaultdict(dict)
                relevant_docs[domain]["test"][qid].update({doc_id: 1})

        corpus = datasets.DatasetDict(corpus)
        queries = datasets.DatasetDict(queries)
        relevant_docs = datasets.DatasetDict(relevant_docs)
        return corpus, queries, relevant_docs


    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = self.load_ifir_data(
            path=self.metadata_dict["dataset"]["path"],
            domains=DOMAINS,
            eval_splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        self.data_loaded = True

