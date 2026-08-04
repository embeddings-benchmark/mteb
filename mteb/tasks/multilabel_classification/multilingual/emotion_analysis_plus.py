from __future__ import annotations

from mteb.abstasks.multilabel_classification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class EmotionAnalysisPlus(AbsTaskMultilabelClassification):
    """
    BRIGHTER-emotion-categories: multi-label emotion detection (28 languages).
    Each sample can express one or more of the six Ekman emotions:
    anger, disgust, fear, joy, sadness, surprise.
    """

    metadata = TaskMetadata(
        name="EmotionAnalysisPlus",
        description=(
            "Multi-label emotion classification dataset for 28 languages "
            "released with the BRIGHTER project and SemEval-2025 Task 11."
        ),
        reference="https://github.com/emotion-analysis-project/SemEval2025-Task11",
        dataset={
            "path": "mteb/EmotionAnalysis",
            "revision": "554dbe305cad4f86705c8b6389c76f7f33fc6fd8",
        },
        type="MultilabelClassification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            # (ISO-639-3 code : BCP-47 tag)
            "afr": ["afr-Latn"],
            "amh": ["amh-Ethi"],
            "arq": ["arq-Arab"],
            "ary": ["ary-Arab"],
            "chn": ["cdo-Hans"],
            "deu": ["deu-Latn"],
            "eng": ["eng-Latn"],
            "esp": ["spa-Latn"],
            "hau": ["hau-Latn"],
            "hin": ["hin-Deva"],
            "ibo": ["ibo-Latn"],
            "ind": ["ind-Latn"],
            "jav": ["jav-Latn"],
            "kin": ["kin-Latn"],
            "mar": ["mar-Deva"],
            "gaz": ["gaz-Latn"],
            "pcm": ["pcm-Latn"],
            "ron": ["ron-Latn"],
            "rus": ["rus-Cyrl"],
            "som": ["som-Latn"],
            "sun": ["sun-Latn"],
            "swh": ["swa-Latn"],
            "swe": ["swe-Latn"],
            "tat": ["tat-Cyrl"],
            "tir": ["tir-Ethi"],
            "ukr": ["ukr-Cyrl"],
            "vmw": ["vmw-Latn"],
            "xho": ["xho-Latn"],
            "yor": ["yor-Latn"],
            "zul": ["zul-Latn"],
        },
        main_score="accuracy",
        date=("2025-01-01", "2025-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Emotion classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
    )
