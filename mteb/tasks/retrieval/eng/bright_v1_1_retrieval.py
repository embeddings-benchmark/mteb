from __future__ import annotations

from collections import defaultdict
from typing import Any

import datasets

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


def load_bright_data(
    path: str,
    domain: str,
    is_long: bool = False,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> RetrievalSplitData:
    examples = datasets.load_dataset(
        path,
        "examples",
        split=domain,
        cache_dir=cache_dir,
        revision=revision,
    )
    queries = examples.select_columns(["id", "query"]).rename_column("query", "text")
    corpus_name = "long_documents" if is_long else "documents"
    gold_ids_field = "gold_ids_long" if is_long else "gold_ids"
    domain_corpus = datasets.load_dataset(
        path, corpus_name, split=domain, cache_dir=cache_dir, revision=revision
    )
    corpus = domain_corpus.select_columns(["id", "content"]).rename_column(
        "content", "text"
    )
    relevant_docs = defaultdict(dict)
    top_ranked = {}
    all_doc_ids = list(corpus["id"])
    have_excluded_ids = False
    for example in examples:
        qid = example["id"]
        for gid in example[gold_ids_field]:
            relevant_docs[qid][gid] = 1
        excluded_ids = example.get("excluded_ids", [])
        if excluded_ids and excluded_ids != ["N/A"]:
            excluded_set = set(excluded_ids)
            top_ranked[qid] = [
                doc_id for doc_id in all_doc_ids if doc_id not in excluded_set
            ]
            have_excluded_ids = True
        else:
            top_ranked[qid] = all_doc_ids
    return RetrievalSplitData(
        corpus=corpus,
        queries=queries,
        relevant_docs=dict(relevant_docs),
        top_ranked=top_ranked if have_excluded_ids else None,
    )


_BIBTEX_CITATION = r"""
@inproceedings{su2024bright,
  archiveprefix = {arXiv},
  author = {Hongjin SU and Howard Yen and Mengzhou Xia and Weijia Shi and Niklas Muennighoff and Han-yu Wang and Liu Haisu and Quan Shi and Zachary S Siegel and Michael Tang and Ruoxi Sun and Jinsung Yoon and Sercan O Arik and Danqi Chen and Tao Yu},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2407.12883},
  title = {{BRIGHT}: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
  url = {https://openreview.net/forum?id=ykuc5q381b},
  year = {2025},
}
"""


class BrightBiologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightBiologyRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Biology StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this biology post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="biology",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightEarthScienceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEarthScienceRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Earth Science StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this earth_science post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="earth_science",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightEconomicsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEconomicsRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Economics StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this economics post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="economics",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightPsychologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPsychologyRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Psychology StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this psychology post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="psychology",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightRoboticsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightRoboticsRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Robotics StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this robotics post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="robotics",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightStackoverflowRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightStackoverflowRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Stack Overflow answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this stackoverflow post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="stackoverflow",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightSustainableLivingRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightSustainableLivingRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Sustainable Living StackExchange answers.",
        type="Retrieval",
        prompt={
            "query": "Represent this sustainable_living post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="sustainable_living",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightPonyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPonyRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of Pony programming language syntax documentation.",
        type="Retrieval",
        prompt={
            "query": "Represent this Pony question for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="pony",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightLeetcodeRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightLeetcodeRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of similar algorithmic problems based on shared solution techniques.",
        type="Retrieval",
        prompt={
            "query": "Represent this Coding problem for searching relevant examples: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="leetcode",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightAopsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightAopsRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of similar Math Olympiad problems from Art of Problem Solving.",
        type="Retrieval",
        prompt={
            "query": "Represent this Math problem for searching relevant examples: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="aops",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightTheoremQATheoremsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightTheoremQATheoremsRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of theorem definitions and proofs from ProofWiki.",
        type="Retrieval",
        prompt={
            "query": "Represent this Math problem for searching relevant theorems: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="theoremqa_theorems",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightTheoremQAQuestionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightTheoremQAQuestionsRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of theorem definitions from ProofWiki given questions rephrased as real-world scenarios.",
        type="Retrieval",
        prompt={
            "query": "Represent this Math problem for searching relevant examples: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="theoremqa_questions",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightBiologyLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightBiologyLongRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Biology StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this biology post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="biology",
                    is_long=True,
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightEarthScienceLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEarthScienceLongRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Earth Science StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this earth_science post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="earth_science",
                    is_long=True,
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightEconomicsLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEconomicsLongRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Economics StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this economics post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="economics",
                    is_long=True,
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightPsychologyLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPsychologyLongRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Psychology StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this psychology post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="psychology",
                    is_long=True,
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightRoboticsLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightRoboticsLongRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Robotics StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this robotics post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="robotics",
                    is_long=True,
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightStackoverflowLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightStackoverflowLongRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Stack Overflow answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this stackoverflow post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="stackoverflow",
                    is_long=True,
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightSustainableLivingLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightSustainableLivingLongRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Sustainable Living StackExchange answers with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this sustainable_living post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="sustainable_living",
                    is_long=True,
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True


class BrightPonyLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPonyLongRetrieval",
        dataset={
            "path": "mteb/BRIGHT",
            "revision": "c26703e6600d97c579ee2985f16cf307db13ed85",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of Pony programming language syntax documentation with long documents.",
        type="Retrieval",
        prompt={
            "query": "Represent this Pony question for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["long"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-03-01", "2024-06-01"),
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": {
                self.eval_splits[0]: load_bright_data(
                    path=self.metadata.dataset["path"],
                    domain="pony",
                    is_long=True,
                    cache_dir=kwargs.get("cache_dir"),
                    revision=self.metadata.dataset["revision"],
                )
            }
        }
        self.data_loaded = True
