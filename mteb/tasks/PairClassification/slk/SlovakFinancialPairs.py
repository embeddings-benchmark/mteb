from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SlovakFinancialPairs(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SlovakFinancialPairs",
        dataset={
            "path": "TUKE-KEMT/slovak-financial-pairs",
            "revision": "d3a2fdfaf58de09485fcee8357e78a151f667cf3",
        },
        description="Question Aswer Pairs from Slovak Financial Exam Corpus",
        reference="",
        category="s2s",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="max_ap",
        date=None,
        domains=["Legal", "Financial"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
""",
        prompt="Odpovedaj na otázku alebo dokonči vetu."
    )

