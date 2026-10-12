from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class UrbanSound8kZeroshotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="UrbanSound8kZeroshot",
        description="Environmental Sound Classification Dataset.",
        reference="https://huggingface.co/datasets/danavery/urbansound8K",
        dataset={
            "path": "mteb/UrbanSound8kZeroshot",
            "revision": "eecb4c1b201d82814db24f2327b703c1fc22b560",
        },
        type="AudioZeroshotClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=["zxx-Zxxx"],
        main_score="accuracy",
        date=("2014-11-01", "2014-11-03"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Classification"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio", "text"],
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
