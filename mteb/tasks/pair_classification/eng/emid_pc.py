from __future__ import annotations

from typing import ClassVar

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import PromptType


class EMIDPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="EMIDPairClassification",
        description=(
            "Pair classification on EMID (Emotionally paired Music and Image "
            "Dataset): determining whether a music clip and an image are "
            "emotionally aligned under EMID's 13-dimension emotion model. "
            "Positive pairs are true audio–image matches from EMID; negatives "
            "pair a clip with an image from a different emotion."
        ),
        reference="https://doi.org/10.1145/3607541.3616821",
        dataset={
            "path": "mteb/EMID-PC-AI",
            "revision": "625b47142af487b5b9566fa6ed7403c0eb5e56e8",
        },
        type="AudioPairClassification",
        category="a2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2023-01-01", "2023-08-01"),
        domains=["Music", "Web"],
        task_subtypes=["Emotion classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{zou2023emid,
  author = {Zou, Jialing and Mei, Jiahao and Ye, Guangze and Huai, Tianyu and Shen, Qiwei and Dong, Daoguo},
  booktitle = {Proceedings of the 1st International Workshop on Multimedia Content Generation and Evaluation: New Methods and Practice},
  doi = {10.1145/3607541.3616821},
  pages = {41--48},
  title = {{EMID}: An Emotional Aligned Dataset in Audio-Visual Modality},
  year = {2023},
}
""",
        is_beta=True,
    )

    input1_column_name: ClassVar[dict[str, str]] = {"audio": "audio"}
    input2_column_name: ClassVar[dict[str, str]] = {"image": "image"}
    label_column_name: str = "label"
    # category a2i → query prepares audio only, document prepares image only
    input1_prompt_type = PromptType.query
    input2_prompt_type = PromptType.document
