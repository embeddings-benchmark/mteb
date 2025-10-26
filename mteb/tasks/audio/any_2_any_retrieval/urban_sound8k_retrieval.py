from mteb.abstasks.image.abs_task_any2any_retrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class UrbanSound8KA2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="UrbanSound8KA2TRetrieval",
        description="UrbanSound8K: Audio-to-text retrieval of urban sound events.",
        reference="https://huggingface.co/datasets/CLAPv2/Urbansound8K",
        dataset={
            "path": "mteb/Urbansound8K_a2t",
            "revision": "b8e64ca746c4798fd35cf85eb9a01500129b1976",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2014-11-01", "2014-11-03"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
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


class UrbanSound8KT2ARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="UrbanSound8KT2ARetrieval",
        description="UrbanSound8K: Text-to-audio retrieval of urban sound events.",
        reference="https://huggingface.co/datasets/CLAPv2/Urbansound8K",
        dataset={
            "path": "mteb/Urbansound8K_t2a",
            "revision": "d50fd1daded60cf136b82d845b9bda65b8b2376e",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2014-11-01", "2014-11-03"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
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
