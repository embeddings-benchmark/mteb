from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class RavdessZeroshotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RavdessZeroshot",
        description="Emotion classification Dataset. RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes neutral,calm, happy, sad, angry, fearful, surprise, and disgust expressions. These 8 emtoions also serve as labels for the dataset.",
        reference="https://huggingface.co/datasets/narad/ravdess",
        dataset={
            "path": "mteb/RavdessZeroshot-retrieval",
            "revision": "f9db49a613bbf7fb0abb6f9779507b27e0b6b8b9",
        },
        type="AudioZeroshotClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-03-01", "2018-03-16"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio", "text"],
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
