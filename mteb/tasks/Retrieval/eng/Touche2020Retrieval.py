from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class Touche2020(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Touche2020",
        description="Touché Task 1: Argument Retrieval for Controversial Questions",
        reference="https://webis.de/events/touche-20/shared-task-1.html",
        dataset={
            "path": "mteb/touche2020",
            "revision": "a34f9a33db75fa0cbb21bb5cfc3dae8dc8bec93f",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@dataset{potthast_2022_6862281,
  author       = {Potthast, Martin and
                  Gienapp, Lukas and
                  Wachsmuth, Henning and
                  Hagen, Matthias and
                  Fröbe, Maik and
                  Bondarenko, Alexander and
                  Ajjour, Yamen and
                  Stein, Benno},
  title        = {{Touché20-Argument-Retrieval-for-Controversial- 
                   Questions}},
  month        = jul,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6862281},
  url          = {https://doi.org/10.5281/zenodo.6862281}
}""",
        n_samples=None,
        avg_character_length={
            "test": {
                "average_document_length": 1719.3347658445412,
                "average_query_length": 43.42857142857143,
                "num_documents": 382545,
                "num_queries": 49,
                "average_relevant_docs_per_query": 19.020408163265305,
            }
        },
    )
