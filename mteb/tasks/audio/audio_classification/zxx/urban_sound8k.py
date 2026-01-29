from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class UrbanSound8kZeroshotClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="UrbanSound8k",
        description="Environmental Sound Classification Dataset.",
        reference="https://huggingface.co/datasets/danavery/urbansound8K",
        dataset={
            "path": "danavery/urbansound8K",
            "revision": "8aa9177a0c5a6949ee4ee4b7fcabb01dfd4ae466",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["train"],
        eval_langs=["zxx-Zxxx"],
        main_score="accuracy",
        date=("2014-11-01", "2014-11-03"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Salamon:UrbanSound:ACMMM:14,
  author = {Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo},
  booktitle = {Proceedings of the 22nd ACM international conference on Multimedia},
  organization = {ACM},
  pages = {1041--1044},
  title = {A Dataset and Taxonomy for Urban Sound Research},
  year = {2014},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "classID"
    is_cross_validation: bool = True


def dataset_transform(self):
    self.dataset = self.stratified_subsampling(
        self.dataset, seed=self.seed, splits=["train"], label=self.label_column_name
    )
