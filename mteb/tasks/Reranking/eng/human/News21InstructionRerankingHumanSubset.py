from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from .....abstasks.AbsTaskReranking import AbsTaskReranking


class News21InstructionRerankingHumanSubset(AbsTaskReranking):
    metadata = TaskMetadata(
        name="News21InstructionRerankingHumanSubset",
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
  title={TREC 2021 News Track Overview},
  author={Soboroff, Ian and Macdonald, Craig and McCreadie, Richard},
  booktitle={TREC},
  year={2021}
}
""",
        prompt="Given a query, rerank the documents by their relevance to the query",
    )
