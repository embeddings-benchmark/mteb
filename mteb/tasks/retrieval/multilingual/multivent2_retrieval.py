from __future__ import annotations

from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/MultiVENT2"
_DATASET_REVISION = "6e13011b7f64d9f11179dc98dc89621694af1fbd"
_REFERENCE = "https://arxiv.org/abs/2410.11619"
_EVAL_LANGS = ["eng-Latn", "ara-Arab", "cmn-Hans", "kor-Hang", "rus-Cyrl", "spa-Latn"]
_BIBTEX = r"""
@misc{kriz2025multivent20massivemultilingual,
  archiveprefix = {arXiv},
  author = {Kriz, Reno and Sanders, Kate and Etter, David and Murray, Kenton and Carpenter, Cameron and Van Ochten, Kelly and Recknor, Hannah and Guallar-Blasco, Jimena and Martin, Alexander and Colaianni, Ronald and King, Nolan and Yang, Eugene and Van Durme, Benjamin},
  eprint = {2410.11619},
  title = {MultiVENT 2.0: A Massive Multilingual Benchmark for Event-Centric Video Retrieval},
  year = {2025},
}
"""
_DESCRIPTION = (
    "MultiVENT 2.0 is an event-centric news video retrieval benchmark: English queries "
    "about specific world events and news videos in Arabic, Chinese, English, Korean, "
    "Russian and Spanish, where relevance often depends on speech or on-screen text. "
    "This is a mini version: all 2,546 test queries and their graded judgments are "
    "kept, but the corpus is reduced from 109,800 videos to the 4,222 judged for any "
    "test query, including the videos judged not relevant. Scores are therefore not "
    "directly comparable to the paper's full-corpus results. "
)


class MultiVENT2MiniT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultiVENT2MiniT2VRetrieval",
        description=_DESCRIPTION + "The corpus is the silent videos.",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        modalities=["text", "video"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2013-01-01", "2024-10-15"),  # estimated: event dates up to the paper
        domains=["News", "Web"],
        task_subtypes=["Event Retrieval"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve news videos about the following event."},
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.dataset.values():
            for split_data in subset.values():
                split_data["corpus"] = split_data["corpus"].select_columns(
                    ["id", "video"]
                )


class MultiVENT2MiniT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultiVENT2MiniT2VARetrieval",
        description=_DESCRIPTION + "The corpus is the videos with their audio tracks.",
        reference=_REFERENCE,
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2va",
        modalities=["text", "video", "audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2013-01-01", "2024-10-15"),  # estimated: event dates up to the paper
        domains=["News", "Web"],
        task_subtypes=["Event Retrieval"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
        prompt={"query": "Retrieve news videos about the following event."},
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.dataset.values():
            for split_data in subset.values():
                split_data["corpus"] = split_data["corpus"].select_columns(
                    ["id", "video", "audio"]
                )
