from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class OKVQAIT2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OKVQAIT2TRetrieval",
        description="Retrieval a Wiki passage to answer query about an image.",
        reference="https://okvqa.allenai.org",
        dataset={
            "path": "izhx/UMRB-OKVQA",
            "revision": "96a84a043f5465893670cf616f90e64086c0417a",
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_10",
        date=("2019-01-01", "2020-07-29"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{marino2019ok,
  author = {Marino, Kenneth and Rastegari, Mohammad and Farhadi, Ali and Mottaghi, Roozbeh},
  booktitle = {Proceedings of the IEEE/cvf conference on computer vision and pattern recognition},
  pages = {3195--3204},
  title = {Ok-vqa: A visual question answering benchmark requiring external knowledge},
  year = {2019},
}
""",
        prompt={
            "query": "Retrieve documents that provide an answer to the question alongside the image."
        },
    )
