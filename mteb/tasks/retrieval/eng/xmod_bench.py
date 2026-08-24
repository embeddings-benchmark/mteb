from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskCategory
    from mteb.types import Modalities

_DATASET = {
    "path": "jupyterjazz/XModBench-MTEB",
    "revision": "db1d4695d359be83e8fa34575970c6d9c58dbfb4",
}
_REFERENCE = "https://arxiv.org/abs/2510.15148"
_CITATION = r"""
@inproceedings{wang2026xmodbench,
  author = {Wang, Xingrui and Liu, Jiang and Huang, Chao and Yu, Xiaodong and Wang, Ze and Sun, Ximeng and Wu, Jialian and Yuille, Alan and Barsoum, Emad and Liu, Zicheng},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  title = {XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models},
  url = {https://arxiv.org/abs/2510.15148},
  year = {2026},
}
"""

_COMMON_METADATA = dict(
    dataset=_DATASET,
    reference=_REFERENCE,
    type="Any2AnyRetrieval",
    eval_splits=["test"],
    main_score="accuracy",
    date=("2025-10-16", "2025-10-16"),
    domains=["AudioScene", "Scene", "Spoken", "Music", "Web"],
    task_subtypes=["Question answering"],
    license="mit",
    annotations_creators="derived",
    dialect=[],
    sample_creation="multiple",
    bibtex_citation=_CITATION,
    is_public=True,
)


def _metadata(
    *,
    name: str,
    direction: str,
    category: TaskCategory,
    modalities: list[Modalities],
    description: str,
) -> TaskMetadata:
    return TaskMetadata(
        name=name,
        description=description,
        category=category,
        modalities=modalities,
        # The direction is a Hub configuration, not a language. A one-entry
        # mapping makes the standard retrieval loader select that config.
        eval_langs={direction: ["eng-Latn", "zho-Hans"]},
        **_COMMON_METADATA,
    )


class XModBenchAT2TRetrieval(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchAT2TRetrieval",
        direction="at2t",
        category="at2t",
        modalities=["audio", "text"],
        description=(
            "XModBench-Lite audio+text-to-text multiple-choice question "
            "answering. Given an audio condition and semantic question, rank "
            "the four source text answers. Uses all 1,000 official A-to-T Lite "
            "items."
        ),
    )


class XModBenchAT2IRetrieval(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchAT2IRetrieval",
        direction="at2i",
        category="at2i",
        modalities=["audio", "text", "image"],
        description=(
            "XModBench-Lite audio+text-to-image multiple-choice question "
            "answering. Given an audio condition and semantic question, rank "
            "the four source image answers. Contains the 617 image items from "
            "the official A-to-V Lite configuration."
        ),
    )


class XModBenchAT2VRetrieval(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchAT2VRetrieval",
        direction="at2v",
        category="at2v",
        modalities=["audio", "text", "video"],
        description=(
            "XModBench-Lite audio+text-to-video multiple-choice question "
            "answering. Given an audio condition and semantic question, rank "
            "the four source video answers. Contains 373 retained video items "
            "from the official A-to-V Lite configuration after excluding ten "
            "questions that reference undecodable source MP4s."
        ),
    )


class XModBenchT2ARetrieval(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchT2ARetrieval",
        direction="t2a",
        category="t2a",
        modalities=["text", "audio"],
        description=(
            "XModBench-Lite text-to-audio multiple-choice question answering. "
            "Given a text condition and semantic question, rank the four source "
            "audio answers. Uses all 1,000 official T-to-A Lite items."
        ),
    )


class XModBenchT2IRetrieval(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchT2IRetrieval",
        direction="t2i",
        category="t2i",
        modalities=["text", "image"],
        description=(
            "XModBench-Lite text-to-image multiple-choice question answering. "
            "Given a text condition and semantic question, rank the four source "
            "image answers. Contains the 617 image items from the official "
            "T-to-V Lite configuration."
        ),
    )


class XModBenchT2VRetrieval(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchT2VRetrieval",
        direction="t2v",
        category="t2v",
        modalities=["text", "video"],
        description=(
            "XModBench-Lite text-to-video multiple-choice question answering. "
            "Given a text condition and semantic question, rank the four source "
            "video answers. Contains 379 retained video items from the official "
            "T-to-V Lite configuration after excluding four questions that "
            "reference unusable source MP4s."
        ),
    )


class XModBenchIT2ARetrieval(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchIT2ARetrieval",
        direction="it2a",
        category="it2a",
        modalities=["image", "text", "audio"],
        description=(
            "XModBench-Lite image+text-to-audio multiple-choice question "
            "answering. Given an image condition and semantic question, rank "
            "the four source audio answers. Contains the 617 image items from "
            "the official V-to-A Lite configuration."
        ),
    )


class XModBenchVT2ARetrieval(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchVT2ARetrieval",
        direction="vt2a",
        category="vt2a",
        modalities=["video", "text", "audio"],
        description=(
            "XModBench-Lite video+text-to-audio multiple-choice question "
            "answering. Given a video condition and semantic question, rank "
            "the four source audio answers. Contains 382 retained video items "
            "from the official V-to-A Lite configuration after excluding one "
            "question that references an undecodable source MP4."
        ),
    )


class XModBenchIT2TRetrieval(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchIT2TRetrieval",
        direction="it2t",
        category="it2t",
        modalities=["image", "text"],
        description=(
            "XModBench-Lite image+text-to-text multiple-choice question "
            "answering. Given an image condition and semantic question, rank "
            "the four source text answers. Contains the 617 image items from "
            "the official V-to-T Lite configuration."
        ),
    )


class XModBenchVT2TRetrieval(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchVT2TRetrieval",
        direction="vt2t",
        category="vt2t",
        modalities=["video", "text"],
        description=(
            "XModBench-Lite video+text-to-text multiple-choice question "
            "answering. Given a video condition and semantic question, rank "
            "the four source text answers. Contains 379 retained video items "
            "from the official V-to-T Lite configuration after excluding four "
            "questions that reference undecodable source MP4s."
        ),
    )


__all__ = [
    "XModBenchAT2IRetrieval",
    "XModBenchAT2TRetrieval",
    "XModBenchAT2VRetrieval",
    "XModBenchIT2ARetrieval",
    "XModBenchIT2TRetrieval",
    "XModBenchT2ARetrieval",
    "XModBenchT2IRetrieval",
    "XModBenchT2VRetrieval",
    "XModBenchVT2ARetrieval",
    "XModBenchVT2TRetrieval",
]
