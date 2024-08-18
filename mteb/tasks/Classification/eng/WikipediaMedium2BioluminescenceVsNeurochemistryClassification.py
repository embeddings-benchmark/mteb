from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaMedium2BioluminescenceVsNeurochemistryClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaMedium2BioluminescenceVsNeurochemistryClassification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/Wikipedia_Medium_2_Class_Bioluminescence_vs_Neurochemistry",
            "revision": "4b1018f7a60702173d5ff9c08fda4704961ca3be",
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
