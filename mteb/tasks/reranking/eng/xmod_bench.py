from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskCategory
    from mteb.types import ISOLanguageScript, Modalities

dataset = {
    "path": "jupyterjazz/XModBench-MTEB",
    "revision": "ab71d2be1ba618cb0f0138231f70b82b3c9a02b3",
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
    dataset=dataset,
    reference=_REFERENCE,
    type="Any2AnyReranking",
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
    languages: list[ISOLanguageScript] | None = None,
) -> TaskMetadata:
    return TaskMetadata(
        name=name,
        description=description,
        category=category,
        modalities=modalities,
        eval_langs={direction: languages if languages is not None else ["eng-Latn"]},
        **_COMMON_METADATA,
    )


class XModBenchAT2TReranking(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchAT2TReranking",
        direction="at2t",
        category="at2t",
        modalities=["audio", "text"],
        languages=["eng-Latn", "zho-Hans"],
        description=(
            "XModBench-Lite audio+text-to-text multiple-choice question "
            "answering. Given an audio condition and semantic question, rank "
            "the four source text answers. Uses all 1,000 official A-to-T Lite "
            "items."
        ),
    )


class XModBenchAT2IReranking(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchAT2IReranking",
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


class XModBenchAT2VReranking(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchAT2VReranking",
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


class XModBenchT2AReranking(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchT2AReranking",
        direction="t2a",
        category="t2a",
        modalities=["text", "audio"],
        description=(
            "XModBench-Lite text-to-audio multiple-choice question answering. "
            "Given a text condition and semantic question, rank the four source "
            "audio answers. Uses all 1,000 official T-to-A Lite items."
        ),
    )


class XModBenchT2IReranking(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchT2IReranking",
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


class XModBenchT2VReranking(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchT2VReranking",
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


class XModBenchIT2AReranking(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchIT2AReranking",
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


class XModBenchVT2AReranking(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchVT2AReranking",
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


class XModBenchIT2TReranking(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchIT2TReranking",
        direction="it2t",
        category="it2t",
        modalities=["image", "text"],
        languages=["eng-Latn", "zho-Hans"],
        description=(
            "XModBench-Lite image+text-to-text multiple-choice question "
            "answering. Given an image condition and semantic question, rank "
            "the four source text answers. Contains the 617 image items from "
            "the official V-to-T Lite configuration."
        ),
    )


class XModBenchVT2TReranking(AbsTaskRetrieval):
    metadata = _metadata(
        name="XModBenchVT2TReranking",
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
