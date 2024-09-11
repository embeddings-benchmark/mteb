from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SDSGlovesClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SDSGlovesClassification",
        description="TBW",
        reference="https://www.kaggle.com/datasets/eliseu10/material-safety-data-sheets",
        dataset={
            "path": "BASF-We-Create-Chemistry/Small-SDS-Gloves-Classification",
            "revision": "31aa32eac2ed1a2a97929d6da722a659e7cc2e2d",
        },
        type="Classification",
        category="s2p",
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
