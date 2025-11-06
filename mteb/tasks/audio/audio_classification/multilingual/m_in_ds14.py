from mteb.abstasks.audio.abs_task_audio_classification import AbsTaskAudioClassification
from mteb.abstasks.task_metadata import TaskMetadata

EVAL_LANGS_MAP = {
    "en-GB": ["eng-Latn"],  # English
    "fr-FR": ["fra-Latn"],  # French
    "it-IT": ["ita-Latn"],  # Italian
    "es-ES": ["spa-Latn"],  # Spanish
    "pt-PT": ["por-Latn"],  # Portuguese
    "de-DE": ["deu-Latn"],  # German
    "nl-NL": ["nld-Latn"],  # Dutch
    "ru-RU": ["rus-Cyrl"],  # Russian
    "pl-PL": ["pol-Latn"],  # Polish
    "cs-CZ": ["ces-Latn"],  # Czech
    "ko-KR": ["kor-Hang"],  # Korean
    "zh-CN": ["zho-Hans"],  # Chinese (Simplified)
}


class MInDS14Classification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="MInDS14",
        description="MInDS-14 is an evaluation resource for intent detection with spoken data in 14 diverse languages.",
        reference="https://arxiv.org/abs/2104.08524",
        dataset={
            "path": "mteb/minds14-multilingual",
            "revision": "58d73d2cddefab7df4bc0b814aa5c53c5fd4928e",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=EVAL_LANGS_MAP,
        main_score="accuracy",
        date=("2021-04-01", "2021-04-30"),  # Paper publication date
        domains=["Speech", "Spoken"],
        task_subtypes=["Intent Classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@article{DBLP:journals/corr/abs-2104-08524,
  author = {Daniela Gerz and Pei{-}Hao Su and Razvan Kusztos and Avishek Mondal and Michal Lis and Eshan Singhal and Nikola Mrkšić and Tsung{-}Hsien Wen and Ivan Vulic},
  eprint = {2104.08524},
  eprinttype = {arXiv},
  journal = {CoRR},
  title = {Multilingual and Cross-Lingual Intent Detection from Spoken Data},
  url = {https://arxiv.org/abs/2104.08524},
  volume = {abs/2104.08524},
  year = {2021},
}
""",
    )

    audio_column_name: str = "audio"
    label_column_name: str = "intent_class"  # Contains numeric labels 0-13
    samples_per_label: int = 40
    is_cross_validation: bool = True
    n_splits: int = 5
