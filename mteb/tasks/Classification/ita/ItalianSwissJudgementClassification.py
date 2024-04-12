from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ItalianSwissJudgementClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ItalianSwissJudgementClassification",
        description="Multilingual, diachronic dataset of Swiss Federal Supreme Court cases annotated with the respective binarized judgment outcome (approval/dismissal)",
        reference="https://aclanthology.org/2021.nllp-1.3/",
        dataset={
            "path": "rcds/swiss_judgment_prediction",
            "revision": "29806f87bba4f23d0707d3b6d9ea5432afefbe2f",
            "language": "it",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=[
            "ita-Latn",
        ],
        main_score="accuracy",
        date=None,
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @InProceedings{niklaus-etal-2021-swiss,
        author = {Niklaus, Joel
                        and Chalkidis, Ilias
                        and Stürmer, Matthias},
        title = {Swiss-Judgment-Prediction: A Multilingual Legal Judgment Prediction Benchmark},
        booktitle = {Proceedings of the 2021 Natural Legal Language Processing Workshop},
        year = {2021},
        location = {Punta Cana, Dominican Republic},
        },
        """,
        n_samples={"train": 3072, "validation": 408, "test": 812},
        avg_character_length=None,
    )
