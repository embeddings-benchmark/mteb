from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaEasy10Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaEasy10Classification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/Wikipedia_Easy_10_Class",
            "revision": "c5ae57cbc4acbf63c49f1b4b1408eb4209e5cf74",
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
