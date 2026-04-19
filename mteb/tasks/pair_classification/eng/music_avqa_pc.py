from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

from ._video_pair_helpers import build_pair_dataset, generate_pairs


class MusicAVQAPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="MusicAVQAPairClassification",
        description=(
            "Pair classification on the MUSIC-AVQA dataset: "
            "determining whether two video clips feature the same "
            "musical instrument from 22 instrument categories."
        ),
        reference="https://gewu-lab.github.io/MUSIC-AVQA/",
        dataset={
            "path": "mteb/MUSIC-AVQA_cls-preprocessed",
            "revision": "29f50ae80ad4e8c1cfdbc0148aefe6fe050833dd",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2022-06-01", "2022-06-30"),
        domains=["Music"],
        task_subtypes=["Music Instrument Recognition"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Li2022Learning,
  author = {Li, Guangyao and Wei, Yake and Tian, Yapeng and Xu, Chenliang and Wen, Ji-Rong and Hu, Di},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {Learning to Answer Questions in Dynamic Audio-Visual Scenarios},
  year = {2022},
}
""",
        contributed_by="stef41",
        is_beta=True,
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        import random

        rng = random.Random(42)
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]
            pairs = generate_pairs(ds["label"], rng)
            self.dataset[split] = build_pair_dataset(ds, pairs)
