from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageTextPairClassification import (
    AbsTaskImageTextPairClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class AROVisualAttribution(AbsTaskImageTextPairClassification):
    images_column_names = ["image"]
    texts_column_names = ["true_caption", "false_caption"]

    metadata = TaskMetadata(
        name="AROVisualAttribution",
        description="Compositionality Evaluation of images to their captions.",
        reference="https://openreview.net/forum?id=KRLUvxh8uaX",
        dataset={
            "path": "gowitheflow/ARO-Visual-Attribution",
            "revision": "18f7e01358d91df599d723f00e16a18640e19398",
        },
        type="Compositionality",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="text_acc",
        date=(
            "2022-01-01",
            "2022-12-31",
        ),  # Estimated range for the collection of data
        domains=["Encyclopaedic"],
        task_subtypes=["Caption Pairing"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{yuksekgonul2023and,
  author = {Yuksekgonul, Mert and Bianchi, Federico and Kalluri, Pratyusha and Jurafsky, Dan and Zou, James},
  booktitle = {The Eleventh International Conference on Learning Representations},
  title = {When and why vision-language models behave like bags-of-words, and what to do about it?},
  year = {2023},
}
""",
        descriptive_stats={
            "n_samples": {"test": 28748},
            "avg_character_length": {"test": 1},
        },
    )
