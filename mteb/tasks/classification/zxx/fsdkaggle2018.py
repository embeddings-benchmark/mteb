from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FSDKaggle2018Classification(AbsTaskClassification):
    """Classify verified FSDKaggle2018 audio clips."""

    metadata = TaskMetadata(
        name="FSDKaggle2018Classification",
        description=(
            "Classify manually verified Freesound clips into 41 AudioSet "
            "sound-event classes."
        ),
        reference="https://arxiv.org/abs/1807.09902",
        dataset={
            "path": "artist/FSDKaggle2018",
            "revision": "f1fa466b5a226463d5b6ba1f8ecbe05f2f74262f",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="accuracy",
        date=("2018-03-30", "2018-07-31"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Classification"],
        # The curated dataset is CC BY 4.0, while each embedded Freesound clip
        # retains its own CC license. Use the authoritative mixed-license notice.
        license="https://zenodo.org/records/2552860",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Fonseca2018_DCASE,
  author = {Fonseca, Eduardo and Plakal, Manoj and Font, Frederic and Ellis, Daniel P. W. and Favory, Xavier and Pons, Jordi and Serra, Xavier},
  booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events 2018 Workshop},
  month = {November},
  pages = {69--73},
  title = {General-purpose Tagging of Freesound Audio with AudioSet Labels: Task Description, Dataset, and Baseline},
  url = {https://arxiv.org/abs/1807.09902},
  year = {2018},
}
""",
        is_beta=True,
    )

    input_column_name = "audio"
    label_column_name = "label"
