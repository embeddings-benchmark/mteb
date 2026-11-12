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
@inproceedings{weller2024followir,
  address = {Albuquerque, New Mexico},
  author = {Weller, Orion and Chang, Benjamin and MacAvaney, Sean and Lo, Kyle and Cohan, Arman and Van Durme, Benjamin and Lawrie, Dawn and Soldaini, Luca},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the {A}mericas Chapter of the {A}ssociation for {C}omputational {L}inguistics: Human Language Technologies (Volume 1: Long Papers)},
  doi = {10.18653/v1/2025.naacl-long.597},
  editor = {Chiruzzo, Luis and Ritter, Alan and Wang, Lu},
  isbn = {979-8-89176-189-6},
  month = {apr},
  pages = {11926--11942},
  publisher = {Association for Computational Linguistics},
  title = {{F}ollow{IR}: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
  url = {https://aclanthology.org/2025.naacl-long.597/},
  year = {2025},
}
""",
        prompt="Given a query, rerank the documents by their relevance to the query",
        adapted_from=["News21InstructionRetrieval"],
    )
