from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

DOMAINS_LONG = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
]

DOMAINS = DOMAINS_LONG + [
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]

DOMAINS_langs = {split: ["eng-Latn"] for split in DOMAINS}


def load_bright_data(
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
            path, "documents", split=domain, cache_dir=cache_dir, revision=revision
        )
        examples = datasets.load_dataset(
            path, "examples", split=domain, cache_dir=cache_dir, revision=revision
        )
        if domain in DOMAINS_LONG:
            domain_corpus_long = datasets.load_dataset(
                path,
                "long_documents",
                split=domain,
                cache_dir=cache_dir,
                revision=revision,
            )
        corpus[domain]["standard"] = {
            e["id"]: {"text": e["content"]} for e in domain_corpus
        }
        if domain in DOMAINS_LONG:
            corpus[domain]["long"] = {
                e["id"]: {"text": e["content"]} for e in domain_corpus_long
            }
        queries[domain]["standard"] = queries[domain]["long"] = {
            e["id"]: e["query"] for e in examples
        }
        relevant_docs[domain]["standard"] = {}
        relevant_docs[domain]["long"] = {}

        for e in examples:
            qid = e["id"]
            gold_ids = e["gold_ids"]
            gold_ids_long = e["gold_ids_long"]
            relevant_docs[domain]["standard"][qid] = defaultdict(dict)
            relevant_docs[domain]["long"][qid] = defaultdict(dict)
            for gid in gold_ids:
                relevant_docs[domain]["standard"][qid].update({gid: 1})
            for gid in gold_ids_long:
                relevant_docs[domain]["long"][qid].update({gid: 1})

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


def load_data(self, **kwargs):
    if self.data_loaded:
        return

    self.corpus, self.queries, self.relevant_docs = self.load_bright_data(
        path=self.metadata_dict["dataset"]["path"],
        domains=DOMAINS,
        eval_splits=self.metadata_dict["eval_splits"],
        cache_dir=kwargs.get("cache_dir", None),
        revision=self.metadata_dict["dataset"]["revision"],
    )
    self.data_loaded = True


class BrightRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Bright retrieval dataset.",
        type="Retrieval",
        category="s2p",
        eval_splits=["standard"],
        eval_langs=DOMAINS_langs,
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=r"""
@misc{su2024brightrealisticchallengingbenchmark,
  archiveprefix = {arXiv},
  author = {Hongjin Su and Howard Yen and Mengzhou Xia and Weijia Shi and Niklas Muennighoff and Han-yu Wang and Haisu Liu and Quan Shi and Zachary S. Siegel and Michael Tang and Ruoxi Sun and Jinsung Yoon and Sercan O. Arik and Danqi Chen and Tao Yu},
  eprint = {2407.12883},
  primaryclass = {cs.CL},
  title = {BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
  url = {https://arxiv.org/abs/2407.12883},
  year = {2024},
}
""",
    )
    load_bright_data = load_bright_data
    load_data = load_data


long_metadata = BrightRetrieval.metadata.model_copy()
long_metadata.eval_splits = ["long"]
long_metadata.description = "Bright retrieval dataset with long documents."
long_metadata.name = "BrightLongRetrieval"

dom_langs_long = {split: ["eng-Latn"] for split in DOMAINS_LONG}
long_metadata.eval_langs = dom_langs_long


class BrightLongRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = long_metadata

    load_bright_data = load_bright_data
    load_data = load_data
