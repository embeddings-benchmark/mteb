from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaEasy2GeneExpressionVsMetallurgyClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaEasy2GeneExpressionVsMetallurgyClassification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/WikipediaEasy2GeneExpressionVsMetallurgyClassification",
            "revision": "6ac491e5de9070c6dd434b31e76d3d379123dcff",
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
