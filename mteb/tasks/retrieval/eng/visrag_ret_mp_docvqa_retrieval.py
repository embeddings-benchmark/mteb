from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VisRAGRetMPDocVQA(AbsTaskRetrieval):
    """VisRAG Retrieval task for MP-DocVQA industrial documents.

    The corpus contains scanned pages from multi-page industrial documents and
    the queries are questions targeting specific pages.  Each query has one
    relevant page image.
    """

    metadata = TaskMetadata(
        name="VisRAGRetMPDocVQA",
        description="Benchmark the ability to retrieve specific relevant pages from multi-page documents and generate answers based on visual evidence.",
        reference="https://doi.org/10.1016/j.patcog.2023.109834",
        type="Retrieval",
        task_subtypes=["Image Text Retrieval"],
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "mteb/VisRAGRetMPDocVQA",
            "revision": "b1feea8bb4b37ade699d6974d71a11868866569f",
        },
        date=("1900-01-01", "2020-12-31"),
        domains=["Web", "Non-fiction"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{tito2023hierarchicalmultimodaltransformersmultipage,
  author = {Tito, Rubèn and Karatzas, Dimosthenis and Valveny, Ernest},
  doi = {10.1016/j.patcog.2023.109834},
  issn = {0031-3203},
  journal = {Pattern Recognition},
  month = {Dec},
  pages = {109834},
  publisher = {Elsevier BV},
  title = {Hierarchical multimodal transformers for Multipage DocVQA},
  url = {https://doi.org/10.1016/j.patcog.2023.109834},
  volume = {144},
  year = {2023},
}""",
    )
