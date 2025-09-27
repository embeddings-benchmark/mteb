from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class Core17InstructionRerankingHumanSubset(AbsTaskReranking):
    metadata = TaskMetadata(
        name="Core17InstructionRerankingHumanSubset",
        description="Human evaluation subset of Core17 instruction retrieval dataset for reranking evaluation.",
        reference="https://trec.nist.gov/data/core2017.html",
        dataset={
            "path": "mteb/mteb-human-core17-reranking",
            "revision": "e2b1a26cb5277a040d7f96a79fef0cf00afe9ffe",
        },
        type="Reranking",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=("2017-01-01", "2017-12-31"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{allan2017trec,
  author = {Allan, James and Harman, Donna and Kanoulas, Evangelos and Li, Dan and Van Gysel, Christophe and Voorhees, Ellen M},
  booktitle = {TREC},
  title = {TREC 2017 Common Core Track Overview},
  year = {2017},
}
""",
        prompt="Given a query, rerank the documents by their relevance to the query",
    )
