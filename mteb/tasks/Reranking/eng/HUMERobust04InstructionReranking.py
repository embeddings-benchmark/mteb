from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class HUMERobust04InstructionReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="HUMERobust04InstructionReranking",
        description="Human evaluation subset of Robust04 instruction retrieval dataset for reranking evaluation.",
        reference="https://trec.nist.gov/data/robust/04.guidelines.html",
        dataset={
            "path": "mteb/mteb-human-robust04-reranking",
            "revision": "77756407fed441d7be778b7464c34ccf4700af2e",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
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
