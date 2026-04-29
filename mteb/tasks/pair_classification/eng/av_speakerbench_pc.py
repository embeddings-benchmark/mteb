from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AVSpeakerBenchPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="AVSpeakerBenchPairClassification",
        description=(
            "Pair classification on AV-SpeakerBench: determining whether "
            "two video clips come from the same source video (same speaker "
            "context) or different source videos (different speakers). "
            "Clips are grouped by their YouTube source video ID, so pairs "
            "from the same source share speakers and visual context."
        ),
        reference="https://arxiv.org/abs/2512.02231",
        dataset={
            "path": "zachz/AV-SpeakerBench-PC",
            "revision": "f27f6a1c6c35be08ccc9305e1a9db0cf9d0621d5",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2025-12-01", "2025-12-31"),
        domains=["Spoken"],
        task_subtypes=["Duplicate Detection"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@article{nguyen2025avspeakerbench,
  author = {Nguyen, Le Thien Phuc and Yu, Zhuoran and Hang, Samuel Low Yu and An, Subin and Lee, Jeongik and Ban, Yohan and Chung, SeungEun and Nguyen, Thanh-Huy and others},
  journal = {arXiv preprint arXiv:2512.02231},
  title = {See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models},
  year = {2025},
}
""",
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"
