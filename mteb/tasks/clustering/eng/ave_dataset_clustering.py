from __future__ import annotations

from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

BIBTEX = r"""
@inproceedings{Tian_2018_ECCV,
  author = {Tian, Yapeng and Shi, Jing and Li, Bochen and Duan, Zhiyao and Xu, Chenliang},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  month = {September},
  title = {Audio-Visual Event Localization in Unconstrained Videos},
  year = {2018},
}
"""

DATASET = {
    "path": "mteb/AVE-Dataset",
    "revision": "f6eb93b4e89456277a242583b5565b801bc1981d",
}

DESCRIPTION_BASE = (
    "Clustering of short synchronized video and audio clips into 28 "
    "sound-event categories from the Audio-Visual Event (AVE) dataset "
    "(Tian et al., ECCV 2018)."
)


class AVEDatasetAudioVideoClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="AVEDatasetAudioVideoClustering",
        description=DESCRIPTION_BASE + " Uses synchronized video and audio.",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/html/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.html",
        dataset=DATASET,
        type="VideoClustering",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2018-01-01", "2018-09-01"),
        domains=["Spoken", "Scene", "Music"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name = ("video", "audio")
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            self.dataset[split] = self.dataset[split].select_columns(
                ["video", "audio", "label"],
            )


class AVEDatasetVideoClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="AVEDatasetVideoClustering",
        description=DESCRIPTION_BASE + " Uses video only.",
        reference="https://openaccess.thecvf.com/content_ECCV_2018/html/Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper.html",
        dataset=DATASET,
        type="VideoClustering",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2018-01-01", "2018-09-01"),
        domains=["Spoken", "Scene", "Music"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        bibtex_citation=BIBTEX,
        is_beta=True,
    )
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "video"
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        for split in self.metadata.eval_splits:
            self.dataset[split] = self.dataset[split].select_columns(
                ["video", "label"],
            )
