from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import AbsTaskZeroShotClassification

CITATION = r"""
@inproceedings{chen2020vggsound,
  author = {Chen, Honglie and Xie, Weidi and Vedaldi, Andrea and Zisserman, Andrew},
  booktitle = {ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  doi = {10.1109/ICASSP40776.2020.9053174},
  organization = {IEEE},
  pages = {721-725},
  title = {VGGSound: A Large-Scale Audio-Visual Dataset},
  year = {2020},
}
"""


class VGGSoundVideoZeroshotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="VGGSoundVideoZeroshot",
        description="VGGSound is a large-scale audio-visual dataset of short YouTube clips spanning 308 sound classes (e.g. 'playing piano', 'dog barking'). Audio is the primary signal. This variant uses both video and audio modalities. This zero-shot classification task predicts the sound class of each video",
        reference="https://arxiv.org/abs/2004.14368",
        dataset={
            "path": "mteb/VGGSound",
            "revision": "a994211ca0996558ff6cbd6977b4c1749f49c889",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-05-04",
            "2020-05-08",
        ),
        domains=["Web"],
        task_subtypes=["Activity recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name = "video"
    label_column_name: str = "label"

    train_split: str = "test"
    is_cross_validation: bool = True

    def get_candidate_labels(self) -> list[str]:
        return self.dataset["test"].features[self.label_column_name].names

    def dataset_transform(self, num_proc=None, **kwargs) -> None:
        self.dataset["test"] = self.dataset["test"].select(range(2048))


class VGGSoundVideoAudioZeroshotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="VGGSoundVideoAudioZeroshot",
        description="VGGSound is a large-scale audio-visual dataset of short YouTube clips spanning 308 sound classes (e.g. 'playing piano', 'dog barking'). Audio is the primary signal. This variant uses both video and audio modalities. This zero-shot classification task predicts the sound class of each video + audio",
        reference="https://arxiv.org/abs/2004.14368",
        dataset={
            "path": "mteb/VGGSound",
            "revision": "a994211ca0996558ff6cbd6977b4c1749f49c889",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-05-04",
            "2020-05-08",
        ),
        domains=["Web"],
        task_subtypes=["Activity recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )

    input_column_name = ("video", "audio")
    label_column_name: str = "label"

    train_split: str = "test"
    is_cross_validation: bool = True

    def get_candidate_labels(self) -> list[str]:
        return self.dataset["test"].features[self.label_column_name].names

    def dataset_transform(self, num_proc=None, **kwargs) -> None:
        self.dataset["test"] = self.dataset["test"].select(range(2048))