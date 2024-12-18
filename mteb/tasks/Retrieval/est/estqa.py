from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class EstQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EstQA",
        dataset={
            "path": "kardosdrur/estonian-qa",
            "revision": "99d6f921d9dd4d09116a6312deceb22c16529cfb",
        },
        description=(
            "EstQA is an Estonian question answering dataset based on Wikipedia."
        ),
        reference="https://www.semanticscholar.org/paper/Extractive-Question-Answering-for-Estonian-Language-182912IAPM-Alum%C3%A4e/ea4f60ab36cadca059c880678bc4c51e293a85d6?utm_source=direct_link",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["est-Latn"],
        main_score="ndcg_at_10",
        date=(
            "2002-08-24",
            "2021-05-10",
        ),  # birth of Estonian Wikipedia to publishing the article
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@mastersthesis{mastersthesis,
  author       = {Anu Käver},
  title        = {Extractive Question Answering for Estonian Language},
  school       = {Tallinn University of Technology (TalTech)},
  year         = 2021
}
""",
    )
