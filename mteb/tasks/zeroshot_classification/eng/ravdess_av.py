from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CITATION = r"""
@article{10.1371/journal.pone.0196391,
  author = {Livingstone, Steven R. AND Russo, Frank A.},
  doi = {10.1371/journal.pone.0196391},
  journal = {PLOS ONE},
  month = {05},
  number = {5},
  pages = {1-35},
  publisher = {Public Library of Science},
  title = {The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English},
  url = {https://doi.org/10.1371/journal.pone.0196391},
  volume = {13},
  year = {2018},
}
"""

DESCRIPTION_BASE = (
    "Emotion zero-shot classification on RAVDESS for 8 emotions: neutral, "
    "calm, happy, sad, angry, fearful, surprise, and disgust expressions."
)


class RAVDESSAVZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RAVDESSAVZeroShot",
        description=DESCRIPTION_BASE
        + " This variant uses both video and audio. Used the metadata train split remapped to audio-visual filenames (~1,440 examples).",
        reference="https://doi.org/10.1371/journal.pone.0196391",
        dataset={
            "path": "mteb/RAVDESSAVZeroShot",
            "revision": "02a545ce207cd34abfedb0a923c9c8340b8d70c6",
        },
        type="VideoZeroshotClassification",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-05-16", "2018-05-16"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-US", "eng-CA"],
        modalities=["video", "audio", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )


class RAVDESSVZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RAVDESSVZeroShot",
        description=DESCRIPTION_BASE
        + " This variant uses video only. Used the metadata train split remapped to audio-visual filenames (~1,440 examples).",
        reference="https://doi.org/10.1371/journal.pone.0196391",
        dataset={
            "path": "mteb/RAVDESSVZeroShot",
            "revision": "28feb00db1dfee21583e960788f85dea15802b04",
        },
        type="VideoZeroshotClassification",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-05-16", "2018-05-16"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-US", "eng-CA"],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=CITATION,
        is_beta=True,
    )
