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
            "path": "llama-lang-adapt/EmotionAnalysisFinal",
            "revision": "9397bb08c58a5591448c44237c6bed258a5d226c",
        },
        type="MultilabelClassification",
        category="t2c",
        modalities=["text"],
        eval_splits=["validation", "test"],
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

    # ---------------------------------------------------------------- constants
    _EMOTION2ID = {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "joy": 3,
        "sadness": 4,
        "surprise": 5,
    }
    _LOOKUP_TEXT = ("text", "sentence", "utterance")
    _LOOKUP_EMO = ("label", "emotions", "emotion", "category", "label_cat")

    # ---------------------------------------------------------------- transform
    def dataset_transform(self, **kwargs) -> None:
        """
        Bring every split to the MTEB expected format:

        * column **text** : sentence/utterance (str)
        * column **label**: list[int] (multi-label IDs 0–5)
        """
        for lang in self.dataset:
            for split in self.dataset[lang]:
                ds = self.dataset[lang][split]

                # ── 1️⃣  locate the text column ───────────────────────────────
                cols = ds.column_names
                text_col = next(c for c in self._LOOKUP_TEXT if c in cols)

                # ── 2️⃣  locate all emotion columns that are present ──────────
                emo_cols = [e for e in self._EMOTION2ID if e in cols]
                if not emo_cols:
                    raise ValueError(
                        f"{lang}/{split}: none of the expected emotion columns "
                        f"{list(self._EMOTION2ID)} were found."
                    )

                # ── 3️⃣  map each row to {text, label} ────────────────────────
                def to_labels(example):
                    labels = [
                        self._EMOTION2ID[emo]      # integer ID
                        for emo in emo_cols        # only the columns that exist
                        if int(example[emo]) == 1  # treat non-zero as “present”
                    ]
                    return {"text": example[text_col], "label": labels}

                ds = ds.map(
                    to_labels,
                    remove_columns=cols,          # drop original columns
                    desc=f"{lang}/{split}",
                )

                # ── 4️⃣  save the cleaned split back ──────────────────────────
                self.dataset[lang][split] = ds

        # ── 5️⃣  make sure every language has a 'train' split ────────────────
        for lang, splits in self.dataset.items():
            if "train" not in splits:
                self.dataset[lang]["train"] = (
                    splits.get("validation")
                    or splits.get("dev")
                    or splits["test"]
                )
