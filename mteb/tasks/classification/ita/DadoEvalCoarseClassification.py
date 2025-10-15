from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class DadoEvalCoarseClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DadoEvalCoarseClassification",
        dataset={
            "path": "MattiaSangermano/DaDoEval",
            "revision": "7a78eb7cc137fdd1c5826be1a9e9813177706509",
        },
        description="The DaDoEval dataset is a curated collection of 2,759 documents authored by Alcide De Gasperi, spanning the period from 1901 to 1954. Each document in the dataset is manually tagged with its date of issue.",
        reference="https://github.com/dhfbk/DaDoEval",
        type="Classification",
        date=("1901-01-01", "1954-12-31"),
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ita-Latn"],
        main_score="accuracy",
        domains=["Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{menini2020dadoeval,
  author = {Menini, Stefano and Moretti, Giovanni and Sprugnoli, Rachele and Tonelli, Sara and others},
  booktitle = {Proceedings of the Seventh Evaluation Campaign of Natural Language Processing and Speech Tools for Italian. Final Workshop (EVALITA 2020)},
  organization = {Accademia University Press},
  pages = {391--397},
  title = {DaDoEval@ EVALITA 2020: Same-genre and cross-genre dating of historical documents},
  year = {2020},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("class", "label")
        unused_cols = [
            col
            for col in self.dataset["test"].column_names
            if col not in ["text", "label"]
        ]
        self.dataset = self.dataset.remove_columns(unused_cols)
