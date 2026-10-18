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
        adapted_from=["News21InstructionRetrieval"],
    )
