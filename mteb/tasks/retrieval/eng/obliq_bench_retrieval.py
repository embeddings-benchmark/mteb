from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET = {
    "path": "mteb/OBLIQBenchRetrieval",
    "revision": "b630be644e0dbdc2fa5f712ae8383ebbb9cbee7e",
}

_CITATION = r"""@article{tchuindjo2026obliq,
  author = {Tchuindjo, Diane and Shah, Devavrat and Khattab, Omar},
  journal = {arXiv preprint arXiv:2605.06235},
  title = {OBLIQ-Bench: Exposing Overlooked Bottlenecks in Modern Retrievers with Latent and Implicit Queries},
  year = {2026},
}"""

_REFERENCE = "https://arxiv.org/abs/2605.06235"

_COMMON_META = dict(
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    main_score="ndcg_at_10",
    date=("2025-01-01", "2026-05-09"),
    task_subtypes=["Reasoning as Retrieval"],
    license="cc-by-4.0",
    annotations_creators="expert-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=_CITATION,
    reference=_REFERENCE,
)


class OBLIQBenchTwitterRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OBLIQBenchTwitterRetrieval",
        description=(
            "Descriptive retrieval over implicit stance: find tweets that "
            "indicate a given geopolitical-conflict stance without expressing "
            "it explicitly. Relevance is evaluated against pooled judgments."
        ),
        dataset=_DATASET,
        eval_langs={"twitter": ["eng-Latn"]},
        domains=["Social"],
        **_COMMON_META,
    )


class OBLIQBenchWildChatRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OBLIQBenchWildChatRetrieval",
        description=(
            "Descriptive retrieval over LLM interaction transcripts: find "
            "Human-AI conversations exhibiting a named behavioral failure "
            "mode. Relevance is evaluated against pooled judgments."
        ),
        dataset=_DATASET,
        eval_langs={"wildchat": ["eng-Latn"]},
        domains=["Web"],
        **_COMMON_META,
    )


class OBLIQBenchMathRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OBLIQBenchMathRetrieval",
        description=(
            "Analogue retrieval via shared reasoning technique: given a math "
            "problem, retrieve problems whose solutions use the same latent "
            "proof strategy across mathematical topics. A per-query "
            "top_ranked candidate set excludes same-source problems before "
            "scoring; relevance is evaluated against pooled judgments."
        ),
        dataset=_DATASET,
        eval_langs={"math": ["eng-Latn"]},
        domains=["Academic"],
        **_COMMON_META,
    )


class OBLIQBenchWritingRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OBLIQBenchWritingRetrieval",
        description=(
            "Analogue retrieval via cross-domain authorship: given a snippet, "
            "retrieve other snippets by the same author across unrelated "
            "topics, forcing stylistic rather than topical match. A per-query "
            "top_ranked candidate set excludes the snippet's source post "
            "before scoring; relevance uses gold authorship labels."
        ),
        dataset=_DATASET,
        eval_langs={"writing": ["eng-Latn"]},
        domains=["Written"],
        **_COMMON_META,
    )


class OBLIQBenchCongressRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OBLIQBenchCongressRetrieval",
        description=(
            "Tip-of-tongue retrieval over congressional hearings: recover a "
            "single obscure passage from a lossy, abstract recollection that "
            "describes the exchange's dynamic while omitting names, dates, "
            "and verbatim phrasing. One gold passage per query."
        ),
        dataset=_DATASET,
        eval_langs={"congress": ["eng-Latn"]},
        domains=["Government"],
        **_COMMON_META,
    )
