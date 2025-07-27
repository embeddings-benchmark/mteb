from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageTextPairClassification import (
    AbsTaskImageTextPairClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class AROFlickrOrder(AbsTaskImageTextPairClassification):
    images_column_names = ["images"]
    texts_column_names = [
        "correct_caption",
        "hard_text_1",
        "hard_text_2",
        "hard_text_3",
        "hard_text_4",
    ]

    metadata = TaskMetadata(
        name="AROFlickrOrder",
        description="Compositionality Evaluation of images to their captions."
        + "Each capation has four hard negatives created by order permutations.",
        reference="https://openreview.net/forum?id=KRLUvxh8uaX",
        dataset={
            "path": "gowitheflow/ARO-Flickr-Order",
            "revision": "1f9485f69c87947812378a1aedf86410c86a0aa8",
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
            "n_samples": {"test": 5000},
            "avg_character_length": {"test": 1},
        },
    )
