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
        reference="https://doi.org/10.1609/aaai.v37i11.26598",
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
        bibtex_citation="""@article{tanaka2023slidevqadatasetdocumentvisual,
  author = {Tanaka, Ryota and Nishida, Kyosuke and Nishida, Kosuke and Hasegawa, Taku and Saito, Itsumi and Saito, Kuniko},
  doi = {10.1609/aaai.v37i11.26598},
  issn = {2159-5399},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  month = {June},
  number = {11},
  pages = {13636–13645},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)},
  title = {SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images},
  url = {https://doi.org/10.1609/aaai.v37i11.26598},
  volume = {37},
  year = {2023},
}""",
    )
