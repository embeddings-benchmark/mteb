from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaHard2BioluminescenceVsLuminescenceClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaHard2BioluminescenceVsLuminescenceClassification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/WikipediaHard2BioluminescenceVsLuminescenceClassification",
            "revision": "21c4dcebe2c5b36a35292e6441e7a10b59bf4896",
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
