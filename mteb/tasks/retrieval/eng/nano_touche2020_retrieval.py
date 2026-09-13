from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoTouche2020Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoTouche2020Retrieval",
        description="NanoTouche2020 is a smaller subset of Touché Task 1: Argument Retrieval for Controversial Questions.",
        reference="https://webis.de/events/touche-20/shared-task-1.html",
        dataset={
            "path": "mteb/NanoTouche2020Retrieval",
            "revision": "ff81c0caff8b6febe1495eae9f2ca3eddad074b8",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-09-23", "2020-09-23"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@dataset{potthast_2022_6862281,
  author = {Potthast, Martin and
Gienapp, Lukas and
Wachsmuth, Henning and
Hagen, Matthias and
Fröbe, Maik and
Bondarenko, Alexander and
Ajjour, Yamen and
Stein, Benno},
  doi = {10.5281/zenodo.6862281},
  month = jul,
  publisher = {Zenodo},
  title = {{Touché20-Argument-Retrieval-for-Controversial-
Questions}},
  url = {https://doi.org/10.5281/zenodo.6862281},
  year = {2022},
}
""",
        prompt={
            "query": "Given a question, retrieve detailed and persuasive arguments that answer the question"
        },
        adapted_from=["Touche2020"],
    )
