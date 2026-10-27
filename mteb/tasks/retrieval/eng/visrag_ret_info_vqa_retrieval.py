from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VisRAGRetInfoVQA(AbsTaskRetrieval):
    """VisRAG Retrieval task for InfoVQA infographics.

    The corpus contains infographic images and the queries are questions about
    those infographics.  Each query points to one relevant infographic image.
    """

    metadata = TaskMetadata(
        name="VisRAGRetInfoVQA",
        description="Evaluate the retrieval and understanding of complex infographics where layout and graphical elements are essential for cross-modal question answering.",
        reference="https://doi.org/10.1109/wacv51458.2022.00264",
        type="Retrieval",
        task_subtypes=["Image Text Retrieval"],
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "mteb/VisRAGRetInfoVQA",
            "revision": "5890a9860a153222b97e641abccb8bcc237722c1",
        },
        date=("2010-01-01", "2020-12-31"),
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{mathew2021infographicvqa,
  author = {Mathew, Minesh and Bagal, Viraj and Tito, Ruben and Karatzas, Dimosthenis and Valveny, Ernest and Jawahar, C. V.},
  booktitle = {2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  doi = {10.1109/wacv51458.2022.00264},
  month = {Jan},
  pages = {2582–2591},
  publisher = {IEEE},
  title = {InfographicVQA},
  url = {https://doi.org/10.1109/wacv51458.2022.00264},
  year = {2022},
}""",
    )
