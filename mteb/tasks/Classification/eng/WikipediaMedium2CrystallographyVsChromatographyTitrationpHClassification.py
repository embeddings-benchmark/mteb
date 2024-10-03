from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification",
            "revision": "740565a6a853aaed1114a13bdfd5fd46857b4f11",
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
