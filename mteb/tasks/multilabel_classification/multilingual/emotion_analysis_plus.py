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
            "split": "test",
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

    # ---------------------------------------------------------------- transform
    def dataset_transform(self, **kwargs) -> None:
        """
        Bring every split to the MTEB expected format:
        * column **text** : sentence/utterance (str)
        * column **label**: list[int] (multi-label IDs 0–5)

        The mteb/EmotionAnalysis mirror (rev 554dbe3) ships data already in
        canonical {text, label} form, and its 'validation' split is declared
        but empty (0 rows). We load test-only (split="test" in the dataset
        metadata), which returns a bare Dataset per language, so we re-wrap
        into a DatasetDict; every split is already {text, label}.
        """
        from datasets import Dataset, DatasetDict

        for lang in self.dataset:
            # split="test" makes load_dataset return a bare Dataset per lang;
            # re-wrap into the split-keyed dict this task expects.
            if isinstance(self.dataset[lang], Dataset):
                self.dataset[lang] = DatasetDict({"test": self.dataset[lang]})
            for split in self.dataset[lang]:
                cols = self.dataset[lang][split].column_names
                if "text" not in cols or "label" not in cols:
                    raise ValueError(
                        f"{lang}/{split}: expected canonical columns "
                        f"{{text, label}}, got {cols}."
                    )
