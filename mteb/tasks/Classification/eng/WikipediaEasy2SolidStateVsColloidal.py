from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaEasy2SolidStateVsColloidal(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaEasy2SolidStateVsColloidal",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/Wikipedia_Easy_2_Class_Solid_State_vs_Colloidal",
            "revision": "c9d4228c53c402cf3d340d3ccbcdb2cc37c8d6f3",
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
