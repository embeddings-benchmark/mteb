from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class Touche2020(AbsTaskRetrieval):
    superseded_by = "Touche2020Retrieval.v3"

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
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-09-23", "2020-09-23"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
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
        prompt={
            "query": "Given a question, retrieve detailed and persuasive arguments that answer the question"
        },
        adapted_from=["Touche2020"],
    )


class Touche2020v3Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Touche2020Retrieval.v3",
        description="Touché Task 1: Argument Retrieval for Controversial Questions",
        reference="https://github.com/castorini/touche-error-analysis",
        dataset={
            "path": "mteb/webis-touche2020-v3",
            "revision": "431886eaecc48f067a3975b70d0949ea2862463c",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-09-23", "2020-09-23"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@INPROCEEDINGS{Thakur_etal_SIGIR2024,
   author = "Nandan Thakur and Luiz Bonifacio and Maik {Fr\"{o}be} and Alexander Bondarenko and Ehsan Kamalloo and Martin Potthast and Matthias Hagen and Jimmy Lin",
   title = "Systematic Evaluation of Neural Retrieval Models on the {Touch\'{e}} 2020 Argument Retrieval Subset of {BEIR}",
   booktitle = "Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval",
   year = 2024,
   address_ = "Washington, D.C."
}""",
        adapted_from=["Touche2020"],
    )
