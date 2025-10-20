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

    for e in examples:
        qid = e["id"]
        gold_ids = e["gold_ids"]
        for gid in gold_ids:
            relevant_docs["standard"][qid].update({gid: 1})

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


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
        description="Bright Biology retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="biology",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
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
        description="Bright Earth Science retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="earth_science",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
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
        description="Bright Economics retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="economics",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
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
        description="Bright Psychology retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="psychology",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
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
        description="Bright Robotics retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="robotics",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
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
        description="Bright Stackoverflow retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="stackoverflow",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
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
        description="Bright Sustainable Living retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="sustainable_living",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
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
        description="Bright Pony retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="pony",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
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
        description="Bright Leetcode retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="leetcode",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
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
        description="Bright Aops retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="aops",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
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
        description="Bright TheoremQA Theorems retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="theoremqa_theorems",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
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
        description="Bright TheoremQA Questions retrieval dataset.",
        type="Retrieval",
        category="s2p",
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

        self.corpus, self.queries, self.relevant_docs = load_bright_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="theoremqa_questions",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
