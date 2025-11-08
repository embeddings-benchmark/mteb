from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class RuClincIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuClincIntentClassification",
        description="Task-oriented dialog systems need to know when a query falls outside their range of supported intents, but current text classification corpora only define label sets that cover every example",
        dataset={
            "path": "DeepPavlov/clinc_oos_ru",
            "revision": "261f1ad4e979b4a9a3099a2f1f6d34c753bf7885",
        },
        reference="https://huggingface.co/datasets/clinc/clinc_oos",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("01-01-2019", "01-01-2019"),
        domains=["Financial", "Web", "Social"],
        task_subtypes=["Intent classification"],
        license="cc-by-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="""""",
    )
