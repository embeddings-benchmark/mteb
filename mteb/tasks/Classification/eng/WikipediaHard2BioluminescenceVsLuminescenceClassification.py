from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaHard2BioluminescenceVsLuminescence(AbsTaskClassification):
    metadata = TaskMetadata(
        name="WikipediaHard2BioluminescenceVsLuminescence",
        description="TBW",
        reference="https://wikipedia.org",
        dataset={
            "path": "BASF-We-Create-Chemistry/Wikipedia_Hard_2_Class_Bioluminescence_vs_Luminescence",
            "revision": "907895e5fe3138626c1c8d8ff26ac90b3c447cf2",
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
