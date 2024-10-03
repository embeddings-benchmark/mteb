from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaEasy2GreenhouseVsEnantiopureClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaEasy2GreenhouseVsEnantiopureClassification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/WikipediaEasy2GreenhouseVsEnantiopureClassification",
            "revision": "0cfc1a83b6ed832454e8f4f93f7a0e26208274d9",
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
