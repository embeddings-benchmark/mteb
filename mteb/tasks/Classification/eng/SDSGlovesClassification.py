from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SDSGlovesClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SDSGlovesClassification",
        description="TBW",
        reference="https://www.kaggle.com/datasets/eliseu10/material-safety-data-sheets",
        dataset={
            "path": "BASF-We-Create-Chemistry/SmallSDSGlovesClassification",
            "revision": "c723236c5ec417d79512e6104aca9d2cd88168f6",
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
