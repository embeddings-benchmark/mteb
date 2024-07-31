from __future__ import annotations

from mteb.abstasks import AbsTaskZeroshotClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CLEVER(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(
        name="CLEVER",
        description="CLEVER.",
        reference="https://openaccess.thecvf.com/content_cvpr_2017/html/Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.html",
        dataset={
            "path": "clip-benchmark/wds_vtab-clevr_closest_object_distance",
            "revision": "ec9c04224a95836ca0344a6000ec8d8bc8a6d4f2",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2016-01-01", "2016-12-20"),
        domains=["Constructed"],
        task_subtypes=["Object recognition"],
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""\
@InProceedings{Johnson_2017_CVPR,
author = {Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens and Fei-Fei, Li and Lawrence Zitnick, C. and Girshick, Ross},
title = {CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}""",
        descriptive_stats={
            "n_samples": {"test": 15000},
            "avg_character_length": {"test": 0},
        },
    )

    image_column_name: str = "webp"
    label_column_name: str = "cls"

    def get_candidate_labels(self) -> list[str]:
        labels = [
            "very nearby",
            "nearby",
            "near",
            "",  # missing this class name in the original dataset: https://huggingface.co/datasets/clip-benchmark/wds_vtab-clevr_closest_object_distance/blob/main/classnames.txt
            "distant",
            "very distant",
        ]

        return [f"{c} shapes." for c in labels]
