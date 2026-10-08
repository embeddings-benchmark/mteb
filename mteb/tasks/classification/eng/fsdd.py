from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class FSDD(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FSDD",
        description="Spoken digit classification of audio into one of 10 classes: 0-9",
        reference="https://huggingface.co/datasets/silky1708/Free-Spoken-Digit-Dataset",
        dataset={
            "path": "mteb/free-spoken-digit-dataset",
            "revision": "c34455c99604d35cb8d27328c267be1478efc903",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2020-10-06"),
        domains=["Music"],
        task_subtypes=["Spoken Digit Classification"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation="",
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"
