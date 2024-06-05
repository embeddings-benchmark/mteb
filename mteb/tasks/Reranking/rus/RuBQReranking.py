from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class RuBQReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="RuBQReranking",
        dataset={
            "path": "ai-forever/rubq-reranking",
            "revision": "2e96b8f098fa4b0950fc58eacadeb31c0d0c7fa2",
        },
        description="Paragraph reranking based on RuBQ 2.0. Give paragraphs that answer the question higher scores.",
        reference="https://openreview.net/pdf?id=P5UQFFoQ4PJ",
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="map",
        date=("2001-01-01", "2021-01-01"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@inproceedings{RuBQ2021,
        title={RuBQ 2.0: An Innovated Russian Question Answering Dataset},
        author={Ivan Rybin and Vladislav Korablinov and Pavel Efimov and Pavel Braslavski},
        booktitle={ESWC},
        year={2021},
        pages={532--547}
        }""",
        n_samples={"test": 1551},
        avg_character_length={"test": 499.9},
    )
