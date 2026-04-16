from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class RAVDESSAVClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RAVDESSAVClassification",
        description="Emotion classification task with video and audio data for 8 emotions: neutral, calm, happy, sad, angry, fearful, surprise, and disgust expressions.",
        reference="https://doi.org/10.1371/journal.pone.0196391",
        dataset={
            "path": "mteb/RAVDESS_AV",
            "revision": "13af08387c3ce5e86c179a3718eb158669268d65",
        },
        type="Classification",
        category="va2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-05-16", "2018-05-16"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-US", "eng-CA"],
        modalities=["video", "audio"],
        sample_creation="found",
        bibtex_citation=r"""
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
""",
    )
    input_column_name = ("video", "audio")
    label_column_name: str = "emotion"
    is_cross_validation = True
    train_split: str = "test"
