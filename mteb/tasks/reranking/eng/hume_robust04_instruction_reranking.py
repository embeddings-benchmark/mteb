from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HUMERobust04InstructionReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HUMERobust04InstructionReranking",
        description="Human evaluation subset of Robust04 instruction retrieval dataset for reranking evaluation.",
        reference="https://trec.nist.gov/data/robust/04.guidelines.html",
        dataset={
            "path": "mteb/HUMERobust04InstructionReranking",
            "revision": "26b83b71e9f8dbd25538a2d6f89a8a1eb856a3eb",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=("2004-01-01", "2004-12-31"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{voorhees2005trec,
  author = {Voorhees, Ellen M},
  booktitle = {TREC},
  title = {TREC 2004 Robust Retrieval Track Overview},
  year = {2005},
}
""",
        prompt="Given a query, rerank the documents by their relevance to the query",
        adapted_from=["Robust04InstructionRetrieval"],
    )
