from __future__ import annotations

from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.tasks.reranking import (
    XModBenchAT2IReranking,
    XModBenchAT2TReranking,
    XModBenchAT2VReranking,
    XModBenchIT2AReranking,
    XModBenchIT2TReranking,
    XModBenchT2AReranking,
    XModBenchT2IReranking,
    XModBenchT2VReranking,
    XModBenchVT2AReranking,
    XModBenchVT2TReranking,
)

_TASKS = [
    XModBenchAT2TReranking(),
    XModBenchAT2IReranking(),
    XModBenchAT2VReranking(),
    XModBenchT2AReranking(),
    XModBenchT2IReranking(),
    XModBenchT2VReranking(),
    XModBenchIT2AReranking(),
    XModBenchVT2AReranking(),
    XModBenchIT2TReranking(),
    XModBenchVT2TReranking(),
]


class XModBench(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="XModBench",
        description=(
            "XModBench-Lite is a unified cross-modal multiple-choice question "
            "answering benchmark spanning text, audio, images, and video. It "
            "retains 5,981 of 6,000 questions across six official "
            "Audio/Vision/Text directions, excluding 19 questions that "
            "reference unusable source MP4s. Each question ranks its four "
            "original answer candidates; the aggregate reports mean accuracy "
            "across ten concrete MTEB modality tasks."
        ),
        reference="https://arxiv.org/abs/2510.15148",
        tasks=_TASKS,
        main_score="accuracy",
        type="Any2AnyReranking",
        modalities=["audio", "image", "text", "video"],
        eval_splits=["test"],
        bibtex_citation=r"""
@inproceedings{wang2026xmodbench,
  author = {Wang, Xingrui and Liu, Jiang and Huang, Chao and Yu, Xiaodong and Wang, Ze and Sun, Ximeng and Wu, Jialian and Yuille, Alan and Barsoum, Emad and Liu, Zicheng},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  title = {XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models},
  url = {https://arxiv.org/abs/2510.15148},
  year = {2026},
}
""",
    )


