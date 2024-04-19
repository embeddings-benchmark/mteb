from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CzechSubjectivityClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CzechSubjectivityClassification",
        description="An Czech dataset for subjectivity classification.",
        reference="https://arxiv.org/abs/2009.08712",
        dataset={
            "path": "pauli31/czech-subjectivity-dataset",
            "revision": "e387ddf167f3eba99936cff89909ed6264f17e1f",
        },
        type="Classification",
        category="s2s",
        date=("2022-04-01", "2022-04-01"),
        eval_splits=["validation", "test"],
        eval_langs=["ces-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{priban-steinberger-2022-czech,
    title = "\{C\}zech Dataset for Cross-lingual Subjectivity Classification",
    author = "P{\v{r}}ib{\'a}{\v{n}}, Pavel  and
      Steinberger, Josef",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.148",
    pages = "1381--1391",
}
""",
        n_samples={"validation": 500, "test": 2000},
        avg_character_length={"validation": 108.2, "test": 108.3},
    )
