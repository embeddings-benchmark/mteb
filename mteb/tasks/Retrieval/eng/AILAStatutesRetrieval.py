from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AILAStatutes(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AILAStatutes",
        description="This dataset is structured for the task of identifying the most relevant statutes for a given situation.",
        reference="https://zenodo.org/records/4063986",
        dataset={
            "path": "mteb/AILA_statutes",
            "revision": "ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e",
        },
        type="Retrieval",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Article retrieval"],
        license="CC BY 4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation="""@dataset{paheli_bhattacharya_2020_4063986,
  author       = {Paheli Bhattacharya and
                  Kripabandhu Ghosh and
                  Saptarshi Ghosh and
                  Arindam Pal and
                  Parth Mehta and
                  Arnab Bhattacharya and
                  Prasenjit Majumder},
  title        = {AILA 2019 Precedent \& Statute Retrieval Task},
  month        = oct,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4063986},
  url          = {https://doi.org/10.5281/zenodo.4063986}
}""",
        n_samples=None,
        avg_character_length={
            "test": {
                "average_document_length": 1973.6341463414635,
                "average_query_length": 3038.42,
                "num_documents": 82,
                "num_queries": 50,
                "average_relevant_docs_per_query": 4.34,
            }
        },
    )
