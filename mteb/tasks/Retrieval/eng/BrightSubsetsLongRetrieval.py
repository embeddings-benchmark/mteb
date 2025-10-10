from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


def load_bright_long_data(
    path: str,
    domain: str,
    eval_splits: list,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    corpus = dict.fromkeys(eval_splits)
    queries = dict.fromkeys(eval_splits)
    relevant_docs = dict.fromkeys(eval_splits)

    domain_corpus_long = datasets.load_dataset(
        path,
        "long_documents",
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
    corpus["long"] = {e["id"]: {"text": e["content"]} for e in domain_corpus_long}
    queries["long"] = {e["id"]: e["query"] for e in examples}
    relevant_docs["long"] = defaultdict(dict)

    for e in examples:
        qid = e["id"]
        gold_ids_long = e["gold_ids_long"]
        for gid in gold_ids_long:
            relevant_docs["long"][qid].update({gid: 1})

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


class BrightBiologyLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightBiologyLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Bright Biology retrieval dataset with long documents.",
        type="Retrieval",
        category="s2p",
        eval_splits=["long"],
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

        self.corpus, self.queries, self.relevant_docs = load_bright_long_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="biology",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightEarthScienceLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEarthScienceLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Bright Earth Science retrieval dataset with long documents.",
        type="Retrieval",
        category="s2p",
        eval_splits=["long"],
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

        self.corpus, self.queries, self.relevant_docs = load_bright_long_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="earth_science",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightEconomicsLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightEconomicsLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Bright Economics retrieval dataset with long documents.",
        type="Retrieval",
        category="s2p",
        eval_splits=["long"],
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

        self.corpus, self.queries, self.relevant_docs = load_bright_long_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="economics",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightPsychologyLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPsychologyLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Bright Psychology retrieval dataset with long documents.",
        type="Retrieval",
        category="s2p",
        eval_splits=["long"],
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

        self.corpus, self.queries, self.relevant_docs = load_bright_long_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="psychology",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightRoboticsLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightRoboticsLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Bright Robotics retrieval dataset with long documents.",
        type="Retrieval",
        category="s2p",
        eval_splits=["long"],
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

        self.corpus, self.queries, self.relevant_docs = load_bright_long_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="robotics",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightStackoverflowLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightStackoverflowLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Bright Stackoverflow retrieval dataset with long documents.",
        type="Retrieval",
        category="s2p",
        eval_splits=["long"],
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

        self.corpus, self.queries, self.relevant_docs = load_bright_long_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="stackoverflow",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightSustainableLivingLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightSustainableLivingLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Bright Sustainable Living retrieval dataset with long documents.",
        type="Retrieval",
        category="s2p",
        eval_splits=["long"],
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

        self.corpus, self.queries, self.relevant_docs = load_bright_long_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="sustainable_living",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightPonyLongRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightPonyLongRetrieval",
        dataset={
            "path": "xlangai/BRIGHT",
            "revision": "a75a0eb483f6a5233a6efc2d63d71540a4443dfb",
        },
        reference="https://huggingface.co/datasets/xlangai/BRIGHT",
        description="Bright Pony retrieval dataset with long documents.",
        type="Retrieval",
        category="s2p",
        eval_splits=["long"],
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

        self.corpus, self.queries, self.relevant_docs = load_bright_long_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="pony",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
