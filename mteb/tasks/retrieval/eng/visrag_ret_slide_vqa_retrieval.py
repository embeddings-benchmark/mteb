from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VisRAGRetSlideVQA(AbsTaskRetrieval):
    """VisRAG Retrieval task for SlideVQA slide decks.

    The corpus contains slide images from educational slide decks and the queries
    are questions about the slides.  Each query has one relevant slide image.
    """

    metadata = TaskMetadata(
        name="VisRAGRetSlideVQA",
        description="Retrieve and reason across multiple slide images within a deck to answer multi-hop questions in a vision-centric retrieval-augmented generation pipeline.",
        reference="https://arxiv.org/abs/2301.04883",
        type="Retrieval",
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "mteb/VisRAGRetSlideVQA",
            "revision": "c62fb65928b0bf7b709cd3084c87edf75a1ba29b",
        },
        date=("2010-01-01", "2022-12-31"),
        domains=["Web"],
        task_subtypes=["Image Text Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{tanaka2023slidevqadatasetdocumentvisual,
  archiveprefix = {arXiv},
  author = {Ryota Tanaka and Kyosuke Nishida and Kosuke Nishida and Taku Hasegawa and Itsumi Saito and Kuniko Saito},
  eprint = {2301.04883},
  primaryclass = {cs.CL},
  title = {SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images},
  url = {https://arxiv.org/abs/2301.04883},
  year = {2023},
}""",
    )
