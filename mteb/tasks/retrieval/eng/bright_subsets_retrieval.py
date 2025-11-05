from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def load_bright_data(
    path: str,
    domain: str,
    eval_splits: list,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    corpus = dict.fromkeys(eval_splits)
    queries = dict.fromkeys(eval_splits)
    relevant_docs = dict.fromkeys(eval_splits)
    top_ranked = dict.fromkeys(eval_splits)

    domain_corpus = datasets.load_dataset(
        path,
        "documents",
        split=domain,
        cache_dir=cache_dir,
        revision=revision,
    )
    examples = datasets.load_dataset(
        path,
        "examples",
        split=domain,
        cache_dir=cache_dir,
        revision=revision,
    )
    corpus["standard"] = {e["id"]: {"text": e["content"]} for e in domain_corpus}
    queries["standard"] = {e["id"]: e["query"] for e in examples}
    relevant_docs["standard"] = defaultdict(dict)
    top_ranked["standard"] = defaultdict(list)

    # Get all document IDs
    all_doc_ids = [e["id"] for e in domain_corpus]

    for e in examples:
        qid = e["id"]
        gold_ids = e["gold_ids"]
        for gid in gold_ids:
            relevant_docs["standard"][qid].update({gid: 1})

        # Create top_ranked: all documents except excluded_ids
        excluded_ids = e.get("excluded_ids", [])
        if excluded_ids and excluded_ids != ["N/A"]:
            excluded_set = set(excluded_ids)
            top_ranked["standard"][qid] = [
                doc_id for doc_id in all_doc_ids if doc_id not in excluded_set
            ]
        else:
            # No exclusions, use all documents
            top_ranked["standard"][qid] = all_doc_ids

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    top_ranked = datasets.DatasetDict(top_ranked)
    return corpus, queries, relevant_docs, top_ranked


_BIBTEX_CITATION = r"""
@misc{su2024brightrealisticchallengingbenchmark,
  archiveprefix = {arXiv},
  author = {Hongjin Su and Howard Yen and Mengzhou Xia and Weijia Shi and Niklas Muennighoff and Han-yu Wang and Haisu Liu and Quan Shi and Zachary S. Siegel and Michael Tang and Ruoxi Sun and Jinsung Yoon and Sercan O. Arik and Danqi Chen and Tao Yu},
  eprint = {2407.12883},
  primaryclass = {cs.CL},
  title = {BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
  url = {https://arxiv.org/abs/2407.12883},
  year = {2024},
}
"""


class BrightBiologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightBiologyRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Biology StackExchange answers.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="biology",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightEarthScienceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEarthScienceRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Earth Science StackExchange answers.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="earth_science",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightEconomicsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEconomicsRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Economics StackExchange answers.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="economics",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightPsychologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPsychologyRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Psychology StackExchange answers.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="psychology",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightRoboticsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightRoboticsRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Robotics StackExchange answers.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="robotics",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightStackoverflowRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightStackoverflowRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Stack Overflow answers.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="stackoverflow",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightSustainableLivingRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightSustainableLivingRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of web documents cited in Sustainable Living StackExchange answers.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="sustainable_living",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightPonyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPonyRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of Pony programming language syntax documentation.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="pony",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightLeetcodeRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightLeetcodeRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of similar algorithmic problems based on shared solution techniques.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="leetcode",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightAopsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightAopsRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of similar Math Olympiad problems from Art of Problem Solving.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="aops",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightTheoremQATheoremsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightTheoremQATheoremsRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of theorem definitions and proofs from ProofWiki.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="theoremqa_theorems",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True


class BrightTheoremQAQuestionsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightTheoremQAQuestionsRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Part of the BRIGHT benchmark for reasoning-intensive retrieval. Retrieval of theorem definitions from ProofWiki given questions rephrased as real-world scenarios.",
        type="Retrieval",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs, self.top_ranked = (
            load_bright_data(
                path=self.metadata.dataset["path"],
                eval_splits=self.metadata.eval_splits,
                domain="theoremqa_questions",
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata.dataset["revision"],
            )
        )
        self.data_loaded = True
