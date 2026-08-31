from __future__ import annotations

from typing import ClassVar

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)

_DATASET_REVISION = "a935fda3cee8a5250746a8d1d6278214d615cc89"
_DATASET_PATH = "pranitchawla/crisismmd-mteb"
_DATE = ("2017-05-31", "2017-11-19")

_CITATION = r"""
@inproceedings{crisismmd2018icwsm,
  author = {Firoj Alam and Ferda Ofli and Muhammad Imran},
  booktitle = {Proceedings of the 12th International AAAI Conference on Web and Social Media},
  title = {{CrisisMMD}: Multimodal Twitter Datasets from Natural Disasters},
  year = {2018},
}

@inproceedings{ofli2020analysis,
  author = {Ferda Ofli and Firoj Alam and Muhammad Imran},
  booktitle = {17th International Conference on Information Systems for Crisis Response and Management},
  title = {Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response},
  year = {2020},
}
"""


class _CrisisMMDZeroShotClassification(AbsTaskZeroShotClassification):
    input_column_name: ClassVar[tuple[str, str]] = ("image", "text")


class CrisisMMDInformativeZeroShotClassification(_CrisisMMDZeroShotClassification):
    metadata = TaskMetadata(
        name="CrisisMMDInformativeZeroShot",
        description=(
            "Classify whether an image and paired English tweet provide information "
            "useful for humanitarian response without labeled examples. Only examples "
            "whose independently annotated image and text labels agree are included."
        ),
        reference="https://arxiv.org/abs/1805.00713",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
            "name": "informative",
        },
        type="ZeroShotClassification",
        category="it2t",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Social", "News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION,
        prompt=(
            "Classify whether the image and tweet contain information useful for "
            "humanitarian response."
        ),
        is_beta=True,
    )

    def get_candidate_labels(self) -> list[str]:  # noqa: PLR6301
        return [
            "an informative disaster-related image and tweet useful for humanitarian response",
            "a non-informative disaster-related image and tweet not useful for humanitarian response",
        ]


class CrisisMMDHumanitarianZeroShotClassification(_CrisisMMDZeroShotClassification):
    metadata = TaskMetadata(
        name="CrisisMMDHumanitarianZeroShot",
        description=(
            "Classify an image and paired English tweet into five humanitarian "
            "information categories without labeled examples. Only examples whose "
            "independently annotated image and text labels agree are included."
        ),
        reference="https://arxiv.org/abs/1805.00713",
        dataset={
            "path": _DATASET_PATH,
            "revision": _DATASET_REVISION,
            "name": "humanitarian",
        },
        type="ZeroShotClassification",
        category="it2t",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Social", "News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION,
        prompt=(
            "Classify the image and tweet by the humanitarian situation they report."
        ),
        is_beta=True,
    )

    def get_candidate_labels(self) -> list[str]:  # noqa: PLR6301
        return [
            "a disaster-related image and tweet about affected, injured, missing, or deceased people",
            "a disaster-related image and tweet about infrastructure and utility damage",
            "a disaster-related image and tweet about rescue, volunteering, or donation efforts",
            "a disaster-related image and tweet containing other relevant humanitarian information",
            "an image and tweet not containing humanitarian information",
        ]
