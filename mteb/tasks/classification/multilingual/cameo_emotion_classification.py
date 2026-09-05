from __future__ import annotations

from typing import ClassVar

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@misc{christop2025cameo,
  author = {Christop, Iwona and Czajka, Maciej},
  eprint = {2505.11051},
  archiveprefix = {arXiv},
  title = {{CAMEO}: Collection of Multilingual Emotional Speech Corpora},
  year = {2025},
}
"""


class CAMEOEmotionClassification(AbsTaskClassification):
    input_column_name: ClassVar[str] = "audio"
    samples_per_label: int = 16
    n_experiments: int = 5

    metadata = TaskMetadata(
        name="CAMEOEmotionClassification",
        description=(
            "Speech emotion recognition across Bengali, English, French, Italian and "
            "Polish, drawn from the CAMEO collection of emotional speech corpora. Every "
            "audio emotion task in mteb is monolingual, so this scores whether an encoder "
            "hears affect independently of the language being spoken. Only the six "
            "emotions common to all five languages are kept, so subsets share a label set. "
            "The source names its splits after the constituent corpora rather than train "
            "and test, so the split here is by speaker, with no speaker in both sides, "
            "since emotion rides on the voice and a shared speaker would let a model match "
            "timbre instead. German, Russian and Spanish are excluded for having one "
            "speaker or none identified, and CREMA-D and RAVDESS are excluded because mteb "
            "already evaluates both separately. Construction script: "
            "scripts/data/cameo_emotion/create_data.py."
        ),
        reference="https://arxiv.org/abs/2505.11051",
        dataset={
            "path": "vnahata/CAMEO-emotion-classification",
            "revision": "a8abf264b656456c255e2435ef5103f6a04525e0",
        },
        type="AudioClassification",
        category="a2c",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs={
            "ben": ["ben-Beng"],
            "eng": ["eng-Latn"],
            "fra": ["fra-Latn"],
            "ita": ["ita-Latn"],
            "pol": ["pol-Latn"],
        },
        main_score="accuracy",
        date=("2023-01-01", "2025-05-16"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )
