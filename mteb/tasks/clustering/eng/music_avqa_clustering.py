from __future__ import annotations

from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class MusicAVQAClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MusicAVQAClustering",
        description=(
            "Clustering of video clips with audio into 22 musical "
            "instrument categories. Extracted from the MUSIC-AVQA "
            "dataset."
        ),
        reference="https://gewu-lab.github.io/MUSIC-AVQA/",
        dataset={
            "path": "mteb/MUSIC-AVQA_cls-preprocessed",
            "revision": "29f50ae80ad4e8c1cfdbc0148aefe6fe050833dd",
        },
        type="VideoClustering",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2022-06-01", "2022-06-30"),
        domains=["Music"],
        task_subtypes=["Thematic clustering"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=r"""
@ARTICLE{Li2022Learning,
    title	= {Learning to Answer Questions in Dynamic Audio-Visual Scenarios},
    author	= {Guangyao li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, Di Hu},
    journal	= {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year	= {2022},
}
""",
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
