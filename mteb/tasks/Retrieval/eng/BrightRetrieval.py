from collections import defaultdict

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

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
    revision: str | None = None,
):
    corpus = {domain: dict.fromkeys(eval_splits) for domain in domains}
    queries = {domain: dict.fromkeys(eval_splits) for domain in domains}
    relevant_docs = {domain: dict.fromkeys(eval_splits) for domain in domains}

    for domain in domains:
        domain_corpus = datasets.load_dataset(
            path, "documents", split=domain, revision=revision
        )
        examples = datasets.load_dataset(
            path, "examples", split=domain, revision=revision
        )
        queries[domain]["standard"] = {e["id"]: e["query"] for e in examples}
        if domain in DOMAINS_LONG and self.is_long:
            domain_corpus_long = datasets.load_dataset(
                path,
                "long_documents",
                split=domain,
                revision=revision,
            )
            corpus[domain]["long"] = {
                e["id"]: {"text": e["content"]} for e in domain_corpus_long
            }
            queries[domain]["long"] = queries[domain]["standard"]
            relevant_docs[domain]["long"] = {}

        corpus[domain]["standard"] = {
            e["id"]: {"text": e["content"]} for e in domain_corpus
        }

        relevant_docs[domain]["standard"] = {}

        for e in examples:
            qid = e["id"]
            gold_ids = e["gold_ids"]
            relevant_docs[domain]["standard"][qid] = defaultdict(dict)
            for gid in gold_ids:
                relevant_docs[domain]["standard"][qid].update({gid: 1})
            if domain in DOMAINS_LONG and self.is_long:
                relevant_docs[domain]["long"][qid] = defaultdict(dict)
                gold_ids_long = e["gold_ids_long"]
                for gid in gold_ids_long:
                    relevant_docs[domain]["long"][qid].update({gid: 1})

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


def load_data(self) -> None:
    if self.data_loaded:
        return

    self.corpus, self.queries, self.relevant_docs = self.load_bright_data(
        path=self.metadata.dataset["path"],
        domains=list(self.metadata.eval_langs.keys()),
        eval_splits=self.metadata.eval_splits,
        revision=self.metadata.dataset["revision"],
    )
    self.data_loaded = True


class BrightRetrieval(AbsTaskRetrieval):
    is_long = False
    metadata = TaskMetadata(
        name="BrightRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Bright retrieval dataset.",
        type="Retrieval",
        category="t2t",
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


class BrightLongRetrieval(AbsTaskRetrieval):
    is_long = True
    metadata = long_metadata

    load_bright_data = load_bright_data
    load_data = load_data
