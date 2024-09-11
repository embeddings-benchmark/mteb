from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class SDSEyeProtectionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SDSEyeProtectionClassification",
        description="TBW",
        reference="https://www.kaggle.com/datasets/eliseu10/material-safety-data-sheets",
        dataset={
            "path": "BASF-We-Create-Chemistry/Small-SDS-Eyes-Protection-Classification",
            "revision": "685c818bc3065dd8974d58656a0072449c032754",
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
