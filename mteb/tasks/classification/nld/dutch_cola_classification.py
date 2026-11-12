from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DutchColaClassification(AbsTaskClassification):
    samples_per_label = 128
    metadata = TaskMetadata(
        name="DutchColaClassification",
        description="Dutch CoLA is a corpus of linguistic acceptability for Dutch.",
        reference="https://huggingface.co/datasets/GroNLP/dutch-cola",
        dataset={
            "path": "clips/mteb-nl-dutch-cola",
            "revision": "2269ed7d95d8abaab829f1592b4b2047372e9f81",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        date=("2024-03-01", "2024-05-01"),
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="f1",
        domains=["Written"],
        task_subtypes=["Linguistic acceptability"],
        license="not specified",  # specified as unknown
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
        prompt="Classificeer de gegeven zin als grammaticaal aanvaardbaar of niet aanvaardbaar",
    )
