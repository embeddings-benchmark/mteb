from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaMedium2CrystallographyVsChromatographyTitrationpHClassification",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/Wikipedia_Medium_2_Class_Crystallography_vs_Chromatography_Titration_pH",
            "revision": "f1b8f8ca2afd4e8e988e077ac7f42aeae1e1a51c",
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
