from __future__ import annotations

from typing import ClassVar

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = [
    "bul-Cyrl",
    "ces-Latn",
    "ell-Grek",
    "est-Latn",
    "hrv-Latn",
    "hun-Latn",
    "lav-Latn",
    "lit-Latn",
    "ron-Latn",
    "slk-Latn",
    "slv-Latn",
    "spa-Latn",
    "tur-Latn",
]

_DATASET_REVISION = "93bfcc674ea6e3837fe3e404ee2f0ebfb2b429ed"


class GLAMI1MMultimodalClassification(AbsTaskClassification):
    input_column_name: ClassVar[tuple[str, str]] = ("image", "text")
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="GLAMI1MMultimodalClassification",
        description=(
            "Classify multilingual fashion products into 191 categories from their "
            "image, product name, and description. The task retains the official "
            "human-labeled test split and uses a deterministic few-shot training pool."
        ),
        reference="https://arxiv.org/abs/2211.14451",
        dataset={
            "path": "artist/glami-1m-mteb",
            "revision": _DATASET_REVISION,
        },
        type="ImageClassification",
        category="it2c",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2022-01-01", "2022-12-31"),
        domains=["E-commerce", "Written"],
        task_subtypes=["Object recognition"],
        license="apache-2.0",
        annotations_creators="automatic-and-reviewed",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{Kosar_2022_BMVC,
  author = {Vaclav Kosar and Antonín Hoskovec and Milan Šulc and Radek Bartyzal},
  booktitle = {33rd British Machine Vision Conference 2022, BMVC 2022, London, UK, November 21-24, 2022},
  publisher = {BMVA Press},
  title = {{GLAMI-1M}: A Multilingual Image-Text Fashion Dataset},
  url = {https://arxiv.org/abs/2211.14451},
  year = {2022},
}
""",
        prompt="Classify the fashion product into its appropriate category.",
    )
