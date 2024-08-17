from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaEasy2GeneExpressionVsMetallurgyClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaEasy2GeneExpressionVsMetallurgyClassification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/Wikipedia_Easy_2_Class_Gene_Expression_vs_Metallurgy",
            "revision": "2a386fa589c865c8bcd6afbf201bc4f871fe9ef6",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation=None,
        bibtex_citation=None,
        descriptive_stats={}
    )
