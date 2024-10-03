from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaMedium2BioluminescenceVsNeurochemistryClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaMedium2BioluminescenceVsNeurochemistryClassification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/WikipediaMedium2BioluminescenceVsNeurochemistryClassification",
            "revision": "2f68b7d34c2be896e46b14533573b366e59e5aae",
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
