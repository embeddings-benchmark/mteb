from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@inproceedings{li2022learning,
  author = {Li, Guangyao and Wei, Yake and Tian, Yapeng and Xu, Chenliang and Wen, Ji-Rong and Hu, Di},
  booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages = {19108--19118},
  title = {Learning to answer questions in dynamic audio-visual scenarios},
  year = {2022},
}
"""


class MusicAVQACLSAudioVideoZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MusicAVQACLSAudioVideoZeroShot",
        description="MUSIC-AVQA classification dataset containing 22 instrument categories. Given a video and audio of someone playing an instrument, the goal is to predict the instrument type. This zero-shot variant uses both video and audio modalities. Filtered the test split to rows with a 22-class instrument answer (~1,706 examples).",
        reference="https://arxiv.org/abs/2203.14072",
        dataset={
            "path": "mteb/MusicAVQACLSAudioVideoZeroShot",
            "revision": "1766b4f0d5817b39c7f53d088bc1451a68ee5430",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2022-06-19"),
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="gpl-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )


class MusicAVQACLSVideoZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MusicAVQACLSVideoZeroShot",
        description="MUSIC-AVQA classification dataset containing 22 instrument categories. Given a video and audio of someone playing an instrument, the goal is to predict the instrument type. This zero-shot variant uses video only. Filtered the test split to rows with a 22-class instrument answer (~1,706 examples).",
        reference="https://arxiv.org/abs/2203.14072",
        dataset={
            "path": "mteb/MusicAVQACLSVideoZeroShot",
            "revision": "f2761f3a89168da572f7c0f1fa35830ed565c4c2",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2022-06-19"),
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="gpl-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )
