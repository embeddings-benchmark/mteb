from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaMedium2ComputationalVsSpectroscopists(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaMedium2ComputationalVsSpectroscopists",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/Wikipedia_Medium_2_Class_Computational_vs_Spectroscopists",
            "revision": "e74c1a94e9a0aca888324e89df2b7086a2f0923f",
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
