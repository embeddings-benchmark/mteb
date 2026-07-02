"""Multi-EuP v2 Slovak classification tasks.

This module implements two classification tasks based on the Multi-EuP v2 corpus,
using native Slovak speeches from the European Parliament to predict:
- Political party affiliation
- Gender of speakers

Note: Uses only speeches originally delivered in Slovak (LANGUAGE=SK) with the full
speech text (TEXT field), not translations.
"""

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@inproceedings{yang-etal-2024-language-bias,
  author = {Yang, Jinrui and Jiang, Fan and Baldwin, Timothy},
  booktitle = {Proceedings of the Fourth Workshop on Multilingual Representation Learning (MRL 2024)},
  doi = {10.18653/v1/2024.mrl-1.23},
  pages = {280--292},
  publisher = {Association for Computational Linguistics},
  title = {Language Bias in Multilingual Information Retrieval: The Nature of the Beast and Mitigation Methods},
  url = {https://aclanthology.org/2024.mrl-1.23/},
  year = {2024},
}
"""


class MultiEupSlovakPartyClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultiEupSlovakPartyClassification",
        description="Multi-class classification task to predict the European Parliament political group from native Slovak speeches in the Multi-EuP v2 corpus. Uses only speeches originally delivered in Slovak.",
        reference="https://aclanthology.org/2024.mrl-1.23/",
        dataset={
            "path": "mteb/MultiEupSlovakPartyClassification",
            "revision": "3758b0758c1ac4af8e36d27858bef579db5d2eee",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2020-01-13", "2024-04-25"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Government", "Spoken"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt="Given a European Parliament deputy utterance as query, find the deputy's political group",
    )


class MultiEupSlovakGenderClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultiEupSlovakGenderClassification",
        description="Binary classification task to predict the gender of Members of the European Parliament from native Slovak speeches in the Multi-EuP v2 corpus. Uses only speeches originally delivered in Slovak.",
        reference="https://aclanthology.org/2024.mrl-1.23/",
        dataset={
            "path": "mteb/MultiEupSlovakGenderClassification",
            "revision": "5990c9386f75401ff518cbc213cdf8292bf123fe",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2020-01-13", "2024-04-25"),
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="accuracy",
        domains=["Government", "Spoken"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt="Given a European Parliament deputy utterance as query, find if the deputy is male or female",
    )
