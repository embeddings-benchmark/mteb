from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NorwegianParliamentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NorwegianParliamentClassification",
        description="Norwegian parliament speeches annotated for sentiment",
        reference="https://huggingface.co/datasets/NbAiLab/norwegian_parliament",
        dataset={
            "path": "NbAiLab/norwegian_parliament",
            "revision": "f7393532774c66312378d30b197610b43d751972",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test", "validation"],
        eval_langs=["nob-Latn"],
        # assumed to be bokmål
        main_score="accuracy",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="",
        n_samples={"test": 1200, "validation": 1200},
        avg_character_length={"test": 1884.0, "validation": 1911.0},
    )
