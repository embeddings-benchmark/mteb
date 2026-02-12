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
        reference="https://arxiv.org/abs/2212.05935",
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
        bibtex_citation="""@misc{tito2023hierarchicalmultimodaltransformersmultipage,
  archiveprefix = {arXiv},
  author = {Rubèn Tito and Dimosthenis Karatzas and Ernest Valveny},
  eprint = {2212.05935},
  primaryclass = {cs.CV},
  title = {Hierarchical multimodal transformers for Multi-Page DocVQA},
  url = {https://arxiv.org/abs/2212.05935},
  year = {2023},
}""",
    )
