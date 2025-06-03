from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


DOMAINS = [
    "Biology",
    "Bioinformatics",
    "Medical-Sciences",
    "MedXpertQA-Exam",
    'MedQA-Diag',
    "PMC-Treatment",
    "PMC-Clinical",
    "IIYi-Clinical",
]

DOMAINS_langs = {split: ["eng-Latn"] for split in DOMAINS}


def load_r2med_data(
    path: str,
    domains: list,
    eval_splits: list,
    cache_dir: str,
    revision: str,
):
    corpus = {domain: {split: None for split in eval_splits} for domain in DOMAINS}
    queries = {domain: {split: None for split in eval_splits} for domain in DOMAINS}
    relevant_docs = {
        domain: {split: None for split in eval_splits} for domain in DOMAINS
    }

    for domain in domains:
        domain_corpus = datasets.load_dataset(
            path, name=domain+"-corpus", split="corpus", cache_dir=cache_dir, revision=revision
        )
        domain_queries = datasets.load_dataset(
            path, name=domain+"-query", split="query", cache_dir=cache_dir, revision=revision
        )
        domain_qrels = datasets.load_dataset(
            path, name=domain+"-qrels", split="qrels", cache_dir=cache_dir, revision=revision
        )
        corpus[domain]["test"] = {
            e["id"]: {"text": e["text"]} for e in domain_corpus
        }
        queries[domain]["test"] = {
            e["id"]: e["text"] for e in domain_queries
        }
        relevant_docs[domain]["test"] = defaultdict(dict)
        for e in domain_qrels:
            qid = e["q_id"]
            pid = e["p_id"]
            relevant_docs[domain]["test"][qid][pid] = int(e["score"])

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


def load_data(self, **kwargs):
    if self.data_loaded:
        return

    self.corpus, self.queries, self.relevant_docs = self.load_r2med_data(
        path=self.metadata_dict["dataset"]["path"],
        domains=DOMAINS,
        eval_splits=self.metadata_dict["eval_splits"],
        cache_dir=kwargs.get("cache_dir", None),
        revision=self.metadata_dict["dataset"]["revision"],
    )
    self.data_loaded = True


class R2MEDRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="R2MEDRetrieval",
        dataset={
            "path": "R2MED/R2MED",
            "revision": "f7cf8ddcc9e5a9c971fa71d7582fab41611c8972",
        },
        reference="https://huggingface.co/R2MED/R2MED",
        description="R2MED retrieval dataset.",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=DOMAINS_langs,
        main_score="ndcg_at_10",
        domains=["Medical", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=r"""
@article{li2025r2med,
  title={R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  author={Li, Lei and Zhou, Xiao and Liu, Zheng},
  journal={arXiv preprint arXiv:2505.14558},
  year={2025}
}
""",
    )
    load_r2med_data = load_r2med_data
    load_data = load_data
