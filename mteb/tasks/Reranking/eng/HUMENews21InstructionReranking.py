from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class HUMENews21InstructionReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="HUMENews21InstructionReranking",
        description="Human evaluation subset of News21 instruction retrieval dataset for reranking evaluation.",
        reference="https://trec.nist.gov/data/news2021.html",
        dataset={
            "path": "mteb/mteb-human-news21-reranking",
            "revision": "22208ecbb54618adb1592fd2ba7cdd92d643d9de",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
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
