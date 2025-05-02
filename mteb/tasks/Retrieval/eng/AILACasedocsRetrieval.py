from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AILACasedocs(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AILACasedocs",
        description="The task is to retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.",
        reference="https://zenodo.org/records/4063986",
        dataset={
            "path": "mteb/AILA_casedocs",
            "revision": "4106e6bcc72e0698d714ea8b101355e3e238431a",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=None,
        sample_creation="found",
        bibtex_citation=r"""
@dataset{paheli_bhattacharya_2020_4063986,
  author = {Paheli Bhattacharya and
Kripabandhu Ghosh and
Saptarshi Ghosh and
Arindam Pal and
Parth Mehta and
Arnab Bhattacharya and
Prasenjit Majumder},
  doi = {10.5281/zenodo.4063986},
  month = oct,
  publisher = {Zenodo},
  title = {AILA 2019 Precedent \& Statute Retrieval Task},
  url = {https://doi.org/10.5281/zenodo.4063986},
  year = {2020},
}
""",
    )
