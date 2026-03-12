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
        reference="https://arxiv.org/abs/2104.12756",
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
        bibtex_citation="""@misc{mathew2021infographicvqa,
  archiveprefix = {arXiv},
  author = {Minesh Mathew and Viraj Bagal and Rubèn Pérez Tito and Dimosthenis Karatzas and Ernest Valveny and C. V Jawahar},
  eprint = {2104.12756},
  primaryclass = {cs.CV},
  title = {InfographicVQA},
  url = {https://arxiv.org/abs/2104.12756},
  year = {2021},
}""",
    )
