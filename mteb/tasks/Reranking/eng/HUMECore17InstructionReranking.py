from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class HUMECore17InstructionReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="HUMECore17InstructionReranking",
        description="Human evaluation subset of Core17 instruction retrieval dataset for reranking evaluation.",
        reference="https://arxiv.org/abs/2403.15246",
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
@misc{weller2024followir,
  archiveprefix = {arXiv},
  author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
  eprint = {2403.15246},
  primaryclass = {cs.IR},
  title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
  year = {2024},
}
""",
        prompt="Given a query, rerank the documents by their relevance to the query",
        adapted_from=["Core17InstructionRetrieval"],
    )
