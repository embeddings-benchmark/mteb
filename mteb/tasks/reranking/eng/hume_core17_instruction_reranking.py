from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HUMECore17InstructionReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HUMECore17InstructionReranking",
        description="Human evaluation subset of Core17 instruction retrieval dataset for reranking evaluation.",
        reference="https://arxiv.org/abs/2403.15246",
        dataset={
            "path": "mteb/HUMECore17InstructionReranking",
            "revision": "3a421a4b0596f6424c2a068d6eee99675ca5e43f",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
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
