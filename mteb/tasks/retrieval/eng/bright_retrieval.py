import warnings

import datasets
from datasets import Dataset

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
    result = {}

    for domain in domains:
        domain_corpus = datasets.load_dataset(
            path, "documents", split=domain, revision=revision
        )
        examples = datasets.load_dataset(
            path, "examples", split=domain, revision=revision
        )

        queries_dict = {e["id"]: e["query"] for e in examples}
        corpus_dict = {e["id"]: {"text": e["content"]} for e in domain_corpus}

        relevant_docs_standard: dict[str, dict[str, int]] = {}
        for e in examples:
            qid = e["id"]
            gold_ids = e["gold_ids"]
            relevant_docs_standard[qid] = {}
            for gid in gold_ids:
                relevant_docs_standard[qid][gid] = 1

        corpus_ds = Dataset.from_list(
            [
                {
                    "id": k,
                    "text": v.get("text", "") if isinstance(v, dict) else v,
                    "title": v.get("title", "") if isinstance(v, dict) else "",
                }
                for k, v in corpus_dict.items()
            ]
        )
        queries_ds = Dataset.from_list(
            [{"id": k, "text": v} for k, v in queries_dict.items()]
        )

        result[domain] = {
            "standard": {
                "corpus": corpus_ds,
                "queries": queries_ds,
                "relevant_docs": relevant_docs_standard,
                "top_ranked": None,
            }
        }

        if domain in DOMAINS_LONG and self.is_long:
            domain_corpus_long = datasets.load_dataset(
                path,
                "long_documents",
                split=domain,
                revision=revision,
            )
            corpus_long_dict = {
                e["id"]: {"text": e["content"]} for e in domain_corpus_long
            }
            corpus_long_ds = Dataset.from_list(
                [
                    {
                        "id": k,
                        "text": v.get("text", "") if isinstance(v, dict) else v,
                        "title": v.get("title", "") if isinstance(v, dict) else "",
                    }
                    for k, v in corpus_long_dict.items()
                ]
            )

            relevant_docs_long: dict[str, dict[str, int]] = {}
            for e in examples:
                qid = e["id"]
                gold_ids_long = e["gold_ids_long"]
                relevant_docs_long[qid] = {}
                for gid in gold_ids_long:
                    relevant_docs_long[qid][gid] = 1

            result[domain]["long"] = {
                "corpus": corpus_long_ds,
                "queries": queries_ds,
                "relevant_docs": relevant_docs_long,
                "top_ranked": None,
            }

    return result


def load_data(self, num_proc: int | None = None, **kwargs) -> None:
    if self.data_loaded:
        return

    warnings.warn(
        "This task contains wrong prompts in the metadata. "
        "Please use BRIGHT(v1.1) benchmark instead.",
        category=DeprecationWarning,
    )

    self.dataset = self.load_bright_data(
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
        description="BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval",
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
@article{su2024bright,
  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
  journal = {arXiv preprint arXiv:2407.12883},
  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
  year = {2024},
}
""",
        superseded_by="BrightBiologyRetrieval",
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
