from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class SardiStanceClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SardiStanceClassification",
        dataset={
            "path": "MattiaSangermano/SardiStance",
            "revision": "e25d91e6f6a28ebef42212128f0d5e275b676233",
        },
        description="SardiStance is a unique dataset designed for the task of stance detection in Italian tweets. It consists of tweets related to the Sardines movement, providing a valuable resource for researchers and practitioners in the field of NLP.",
        reference="https://github.com/mirkolai/evalita-sardistance",
        type="Classification",
        category="t2c",
        date=("2019-11-01", "2020-01-31"),
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["ita-Latn"],
        main_score="accuracy",
        domains=["Social"],
        task_subtypes=["Political classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{cignarella2020sardistance,
  author = {Cignarella, Alessandra Teresa and Lai, Mirko and Bosco, Cristina and Patti, Viviana and Rosso, Paolo and others},
  booktitle = {CEUR WORKSHOP PROCEEDINGS},
  organization = {Ceur},
  pages = {1--10},
  title = {Sardistance@ evalita2020: Overview of the task on stance detection in italian tweets},
  year = {2020},
}
""",
    )

    def dataset_transform(self):
        unused_cols = [
            col
            for col in self.dataset["test"].column_names
            if col not in ["text", "label"]
        ]
        self.dataset = self.dataset.remove_columns(unused_cols)
