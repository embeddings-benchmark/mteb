from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

from .human_animal_cartoon_pc import _build_pair_dataset, _generate_pairs


class AVEDatasetPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="AVEDatasetPairClassification",
        description=(
            "Pair classification on the Audio-Visual Event (AVE) dataset: "
            "determining whether two video clips contain the same "
            "audio-visual event from 28 categories "
            "(e.g. accordion, guitar, helicopter, speech)."
        ),
        reference="https://openaccess.thecvf.com/content_ECCV_2018/html/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.html",
        dataset={
            "path": "mteb/AVE-Dataset",
            "revision": "f6eb93b4e89456277a242583b5565b801bc1981d",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2018-01-01", "2018-09-01"),
        domains=["Spoken", "Scene", "Music"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Tian_2018_ECCV,
  author = {Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  month = {September},
  title = {Audio-Visual Event Localization in Unconstrained Videos},
  year = {2018},
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
            pairs = _generate_pairs(ds["label"], rng)
            self.dataset[split] = _build_pair_dataset(ds, pairs)
