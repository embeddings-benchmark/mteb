from mteb.abstasks.multilabel_classification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class EmitClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="EmitClassification",
        description="The EMit dataset is a comprehensive resource for the detection of emotions in Italian social media texts. The EMit dataset consists of social media messages about TV shows, TV series, music videos, and advertisements. Each message is annotated with one or more of the 8 primary emotions defined by Plutchik (anger, anticipation, disgust, fear, joy, sadness, surprise, trust), as well as an additional label “love.”",
        reference="https://github.com/oaraque/emit",
        dataset={
            "path": "MattiaSangermano/emit",
            "revision": "b0ceff2da0ca463d5c8c97a4e1c6e40545a1c3a6",
        },
        type="MultilabelClassification",
        category="t2c",
        modalities=["text"],
        date=("2022-01-01", "2022-12-31"),
        eval_splits=["test"],
        eval_langs=["ita-Latn"],
        main_score="accuracy",
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{araque2023emit,
  author = {Araque, O and Frenda, S and Sprugnoli, R and Nozza, D and Patti, V and others},
  booktitle = {CEUR WORKSHOP PROCEEDINGS},
  organization = {CEUR-WS},
  pages = {1--8},
  title = {EMit at EVALITA 2023: Overview of the Categorical Emotion Detection in Italian Social Media Task},
  volume = {3473},
  year = {2023},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"emotion_labels": "label"})
        unused_cols = [
            col
            for col in self.dataset["test"].column_names
            if col not in ["text", "label"]
        ]
        self.dataset = self.dataset.remove_columns(unused_cols)
