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
            "path": "KocLab-Bilkent/turkish-constitutional-court",
            "revision": "main",
        },
        date=("2000-01-01", "2023-02-20"),  # dataset card last updated Feb 20, 2023
        domains=["Legal", "Non-fiction"],
        task_subtypes=["Political classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @article{mumcuoglu21natural,
        title = {{Natural language processing in law: Prediction of outcomes in the higher courts of Turkey}},
        journal = {Information Processing \\& Management},
        volume = {58},
        number = {5},
        year = {2021},
        author = {Mumcuoğlu, Emre and Öztürk, Ceyhun E. and Ozaktas, Haldun M. and Koç, Aykut}
        }
        """,
    )

    def load_data(self, **kwargs):
        # Let the parent load the HF dataset first
        super().load_data(**kwargs)

        # Normalize column names and cast labels to integers for sklearn metrics.
        label_map = {"No violation": 0, "Violation": 1}

        def rename_and_cast(split):
            if "Text" in split.column_names:
                split = split.rename_column("Text", "text")
            if "Label" in split.column_names:
                split = split.rename_column("Label", "label")
            return split.map(lambda ex: {"label": label_map[ex["label"]]})

        if isinstance(self.dataset, dict):
            self.dataset = {
                name: rename_and_cast(split) for name, split in self.dataset.items()
            }
        else:
            self.dataset = rename_and_cast(self.dataset)
