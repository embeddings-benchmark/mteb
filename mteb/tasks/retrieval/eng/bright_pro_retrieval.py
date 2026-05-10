from __future__ import annotations

from collections import defaultdict

import datasets

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def load_bright_pro_data(
    path: str,
    domain: str,
    eval_splits: list,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    """Load Bright-Pro retrieval data for a given StackExchange domain.

    The HuggingFace dataset uses three configs (`documents`, `examples`,
    `aspects`), each split by domain. We only need `documents` (corpus) and
    `examples` (queries + binary gold labels) for standard retrieval. The
    `aspects` config carries fine-grained reasoning-aspect annotations
    (weighted supporting docs); these are not consumed by standard nDCG@10
    but remain available on the Hub for users who want aspect-aware metrics.
    """
    eval_split = eval_splits[0]

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

    corpus[eval_split] = {e["id"]: {"text": e["content"]} for e in domain_corpus}
    # examples["id"] is int64 in the HF schema; coerce to str for consistency
    # with corpus doc-ids (which are strings) and with MTEB's qid conventions.
    queries[eval_split] = {str(e["id"]): e["query"] for e in examples}
    relevant_docs[eval_split] = defaultdict(dict)

    for e in examples:
        qid = str(e["id"])
        for gid in e["gold_ids"]:
            relevant_docs[eval_split][qid][gid] = 1

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


_BIBTEX_CITATION = r"""
@article{Zhao2026RethinkingRR,
  author = {Yilun Zhao and Jinbiao Wei and Tingyu Song and Siyue Zhang and Chen Zhao and Arman Cohan},
  journal = {arXiv preprint arXiv:2605.04018},
  title = {Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems},
  year = {2026},
}
"""


_DATASET_PATH = "yale-nlp/Bright-Pro"
_DATASET_REVISION = "dbdc22babbef310210e267b99249e7cec86d5edf"
_REFERENCE = "https://huggingface.co/datasets/yale-nlp/Bright-Pro"
_DATE = ("2025-09-01", "2026-04-30")


class BrightProBiologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProBiologyRetrieval",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
        },
        reference=_REFERENCE,
        description=(
            "Part of the BRIGHT-Pro benchmark for reasoning-intensive retrieval "
            "in agentic search settings. Biology StackExchange queries are paired "
            "with multi-aspect gold evidence: each query has a long-form reference "
            "answer whose cited passages collectively cover several reasoning "
            "aspects. This task evaluates standard top-k retrieval over the union "
            "of those gold passages."
        ),
        type="Retrieval",
        prompt={
            "query": "Represent this biology post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_bright_pro_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="biology",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightProEarthScienceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProEarthScienceRetrieval",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
        },
        reference=_REFERENCE,
        description=(
            "Part of the BRIGHT-Pro benchmark for reasoning-intensive retrieval "
            "in agentic search settings. Earth Science StackExchange queries are "
            "paired with multi-aspect gold evidence drawn from a long-form "
            "reference answer covering several reasoning aspects."
        ),
        type="Retrieval",
        prompt={
            "query": "Represent this earth_science post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_bright_pro_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="earth_science",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightProEconomicsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProEconomicsRetrieval",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
        },
        reference=_REFERENCE,
        description=(
            "Part of the BRIGHT-Pro benchmark for reasoning-intensive retrieval "
            "in agentic search settings. Economics StackExchange queries are "
            "paired with multi-aspect gold evidence drawn from a long-form "
            "reference answer covering several reasoning aspects."
        ),
        type="Retrieval",
        prompt={
            "query": "Represent this economics post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_bright_pro_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="economics",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightProPsychologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProPsychologyRetrieval",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
        },
        reference=_REFERENCE,
        description=(
            "Part of the BRIGHT-Pro benchmark for reasoning-intensive retrieval "
            "in agentic search settings. Psychology StackExchange queries are "
            "paired with multi-aspect gold evidence drawn from a long-form "
            "reference answer covering several reasoning aspects."
        ),
        type="Retrieval",
        prompt={
            "query": "Represent this psychology post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_bright_pro_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="psychology",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightProRoboticsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProRoboticsRetrieval",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
        },
        reference=_REFERENCE,
        description=(
            "Part of the BRIGHT-Pro benchmark for reasoning-intensive retrieval "
            "in agentic search settings. Robotics StackExchange queries are "
            "paired with multi-aspect gold evidence drawn from a long-form "
            "reference answer covering several reasoning aspects."
        ),
        type="Retrieval",
        prompt={
            "query": "Represent this robotics post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_bright_pro_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="robotics",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightProStackoverflowRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProStackoverflowRetrieval",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
        },
        reference=_REFERENCE,
        description=(
            "Part of the BRIGHT-Pro benchmark for reasoning-intensive retrieval "
            "in agentic search settings. Stack Overflow queries are paired with "
            "multi-aspect gold evidence drawn from a long-form reference answer "
            "covering several reasoning aspects."
        ),
        type="Retrieval",
        prompt={
            "query": "Represent this stackoverflow post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_bright_pro_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="stackoverflow",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True


class BrightProSustainableLivingRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProSustainableLivingRetrieval",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
        },
        reference=_REFERENCE,
        description=(
            "Part of the BRIGHT-Pro benchmark for reasoning-intensive retrieval "
            "in agentic search settings. Sustainable Living StackExchange queries "
            "are paired with multi-aspect gold evidence drawn from a long-form "
            "reference answer covering several reasoning aspects."
        ),
        type="Retrieval",
        prompt={
            "query": "Represent this sustainable_living post for searching relevant passages: "
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_bright_pro_data(
            path=self.metadata.dataset["path"],
            eval_splits=self.metadata.eval_splits,
            domain="sustainable_living",
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata.dataset["revision"],
        )
        self.data_loaded = True
