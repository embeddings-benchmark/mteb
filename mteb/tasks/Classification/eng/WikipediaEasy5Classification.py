from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaEasy5Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaEasy5Classification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/Wikipedia_Easy_5_Class",
            "revision": "14df6a56f71a288622c63a473e81f9205af8e1a7",
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
