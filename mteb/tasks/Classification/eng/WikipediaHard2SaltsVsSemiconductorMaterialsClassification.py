from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaHard2SaltsVsSemiconductorMaterialsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaHard2SaltsVsSemiconductorMaterialsClassification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/WikipediaHard2SaltsVsSemiconductorMaterialsClassification",
            "revision": "9e5415a096012fa2d1f3a929952cf9859e4550e7",
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
