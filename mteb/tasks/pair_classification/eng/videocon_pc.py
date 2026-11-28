from __future__ import annotations

from typing import ClassVar

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class VideoConPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="VideoConPairClassification",
        description=(
            "Pair classification on the VideoCon dataset: "
            "determining whether a text caption correctly describes "
            "a video or is a semantically-plausible contrast caption "
            "(e.g. entity/action/attribute swaps, event order flips). "
            "Tests video-language alignment robustness."
        ),
        reference="https://doi.org/10.1109/cvpr52733.2024.01321",
        dataset={
            "path": "zachz/VideoCon-PC",
            "revision": "b0cb7112a3edfccf8ae697602442a828712b056b",
        },
        type="VideoPairClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2023-11-01", "2023-11-30"),
        domains=["Activity", "Web"],
        task_subtypes=["Caption Pairing"],
        license="mit",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="LM-generated and verified",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{bansal2023videocon,
  author = {Bansal, Hritik and Bitton, Yonatan and Szpektor, Idan and Chang, Kai-Wei and Grover, Aditya},
  booktitle = {2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  doi = {10.1109/cvpr52733.2024.01321},
  month = {June},
  pages = {13927–13937},
  publisher = {IEEE},
  title = {VideoCon: Robust Video-Language Alignment via Contrast Captions},
  url = {https://doi.org/10.1109/cvpr52733.2024.01321},
  year = {2024},
}
""",
    )

    input1_column_name: ClassVar[dict[str, str]] = {"video": "video"}
    input2_column_name: ClassVar[dict[str, str]] = {"text": "text"}
    label_column_name: str = "label"
