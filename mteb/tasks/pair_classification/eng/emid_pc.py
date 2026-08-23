from __future__ import annotations

from typing import ClassVar

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EMIDPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="EMIDPairClassification",
        description=(
            "Pair classification on EMID: determining whether a music clip and an "
            "image share the same emotion category. Positive pairs are true "
            "audio–image matches from EMID; negatives pair a clip with an image "
            "from a different emotion."
        ),
        reference="https://arxiv.org/abs/2308.07622",
        dataset={
            "path": "Wissam42/EMID-PC-AI",
            "revision": "7758c20e79039527e717939e1c7590777f989489",
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
        sample_creation="found",
        bibtex_citation=r"""
@article{zhang2023emid,
  author = {Zhang, Yujie and others},
  journal = {arXiv preprint arXiv:2308.07622},
  title = {Emotionally Paired Music and Image Dataset},
  year = {2023},
}
""",
        is_beta=True,
    )

    input1_column_name: ClassVar[dict[str, str]] = {"audio": "audio"}
    input2_column_name: ClassVar[dict[str, str]] = {"image": "image"}
    label_column_name: str = "label"
