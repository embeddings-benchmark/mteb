from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AESDD(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AESDD",
        description="Acted Emotional Speech Dynamic Database (AESDD): emotion classification of acted Greek speech into one of 5 classes: anger, disgust, fear, happiness, and sadness.",
        reference="https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/",
        dataset={
            "path": "mteb/AESDD-fixed",
            "revision": "2bfe310368859dadb0dbbcc8874135f817f469df",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["train"],
        eval_langs=["ell-Grek"],
        main_score="accuracy",
        date=("2017-10-01", "2017-10-31"),
        domains=["Speech"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@article{vryzas2018speech,
  author = {Vryzas, Nikolaos and Kotsakis, Rigas and Liatsou, Aikaterini and Dimoulas, Charalampos A and Kalliris, George},
  journal = {Journal of the Audio Engineering Society},
  number = {6},
  pages = {457--467},
  publisher = {Audio Engineering Society},
  title = {Speech emotion recognition for performance interaction},
  volume = {66},
  year = {2018},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"

    is_cross_validation: bool = True
