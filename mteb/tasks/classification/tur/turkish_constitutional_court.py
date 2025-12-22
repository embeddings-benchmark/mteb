from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class TurkishConstitutionalCourtViolation(AbsTaskClassification):
    # Normalize column names after load_data renames them.
    label_column_name = "label"
    input_column_name = "text"

    metadata = TaskMetadata(
        name="TurkishConstitutionalCourtViolation",
        description="Binary classification of Turkish constitutional court decisions: Violation vs No violation.",
        reference="https://huggingface.co/datasets/KocLab-Bilkent/turkish-constitutional-court",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="f1",
        dataset={
            "path": "denizgulal/turkish-constitutional-court-violation-clean",
            "revision": "333f49b7ddc72fa4a86ec5bd756a28c585311c74",
        },
        date=("2000-01-01", "2023-02-20"),  # dataset card last updated Feb 20, 2023
        domains=["Legal", "Non-fiction"],
        task_subtypes=["Political classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{mumcuoglu2021natural,
  author = {Mumcuoglu, Emre and Ozturk, Ceyhun E. and Ozaktas, Haldun M. and Koc, Aykut},
  journal = {Information Processing and Management},
  number = {5},
  title = {Natural language processing in law: Prediction of outcomes in the higher courts of Turkey},
  volume = {58},
  year = {2021},
}
""",
    )
