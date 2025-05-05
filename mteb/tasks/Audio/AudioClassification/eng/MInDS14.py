from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import AbsTaskAudioClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class MInDS14Classification(AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="MInDS14",
        description="MInDS-14 is an evaluation resource for intent detection with spoken data in 14 diverse languages.",
        reference="https://arxiv.org/abs/2104.08524",
        dataset={
            "path": "PolyAI/all",
            "name": "all",  # All languages
            "revision": "75900a7c6f93f014f25b50d16596a6da89add3a5",
        },
        type="AudioClassification",
        category="a2t",
        eval_splits=["train"],
        eval_langs=[
            "eng-Latn",  # English
            "fra-Latn",  # French
            "ita-Latn",  # Italian
            "spa-Latn",  # Spanish
            "por-Latn",  # Portuguese
            "deu-Latn",  # German
            "nld-Latn",  # Dutch
            "rus-Cyrl",  # Russian
            "pol-Latn",  # Polish
            "ces-Latn",  # Czech
            "kor-Hang",  # Korean
            "zho-Hans",  # Chinese (Simplified)
        ],
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
        descriptive_stats={
            "n_samples": {
                "train": 563,  # Count for en-US configuration, rest are roughly similar
            },
            "n_classes": 14,
            "classes": [
                "aboard",
                "address",
                "app_error",
                "atm_limit",
                "balance",
                "business_loan",
                "card_issues",
                "cash_deposite",
                "direct_debit",
                "freeze",
                "latest_transactions",
                "joint_account",
                "high_value_payment",
                "pay_bill",
            ],
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "intent_class"  # Contains numeric labels 0-13
    samples_per_label: int = 40
    is_cross_validation: bool = True
    n_splits: int = 5
