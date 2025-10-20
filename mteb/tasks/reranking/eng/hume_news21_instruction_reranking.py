from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HUMENews21InstructionReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HUMENews21InstructionReranking",
        description="Human evaluation subset of News21 instruction retrieval dataset for reranking evaluation.",
        reference="https://trec.nist.gov/data/news2021.html",
        dataset={
            "path": "mteb/HUMENews21InstructionReranking",
            "revision": "be28ab67f72893dd2aeeaef7e2d1c29b44d4495f",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=("2021-01-01", "2021-12-31"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{soboroff2021trec,
  author = {Soboroff, Ian and Macdonald, Craig and McCreadie, Richard},
  booktitle = {TREC},
  title = {TREC 2021 News Track Overview},
  year = {2021},
}
""",
        prompt="Given a query, rerank the documents by their relevance to the query",
        adapted_from=["News21InstructionRetrieval"],
    )
