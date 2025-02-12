from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class MInDS14EnClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="MInDS14EnClustering",
        description="MINDS-14 is a dataset for intent detection in e-banking, covering 14 intents across 14 languages.",
        dataset={
            "path": "FinanceMTEB/MInDS-14-en",
            "revision": "141ac6a9010b851452a9327edfda190d37399b15",
        },
        reference="https://arxiv.org/pdf/2104.08524",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2014-12-01", "2021-03-31"),
        domains=["Finance"],
        license="not specified",
        annotations_creators="expert-annotated",
        bibtex_citation="""@misc{gerz2021multilingualcrosslingualintentdetection,
              title={Multilingual and Cross-Lingual Intent Detection from Spoken Data},
              author={Daniela Gerz and Pei-Hao Su and Razvan Kusztos and Avishek Mondal and Michał Lis and Eshan Singhal and Nikola Mrkšić and Tsung-Hsien Wen and Ivan Vulić},
              year={2021},
              eprint={2104.08524},
              archivePrefix={arXiv},
              primaryClass={cs.CL},
              url={https://arxiv.org/abs/2104.08524},
        }""",
        descriptive_stats={
            "num_samples": {"test": 182},
            "average_text_length": {"test": 522.7857142857143},
            "average_labels_per_text": {"test": 522.7857142857143},
            "unique_labels": {"test": 14},
            "labels": {
                "test": {
                    "BALANCE": {"count": 6929},
                    "PAY_BILL": {"count": 6929},
                    "FREEZE": {"count": 7605},
                    "APP_ERROR": {"count": 7098},
                    "ADDRESS": {"count": 5746},
                    "ATM_LIMIT": {"count": 6929},
                    "ABROAD": {"count": 5746},
                    "DIRECT_DEBIT": {"count": 6084},
                    "LATEST_TRANSACTIONS": {"count": 5746},
                    "HIGH_VALUE_PAYMENT": {"count": 6760},
                    "CASH_DEPOSIT": {"count": 8112},
                    "CARD_ISSUES": {"count": 7774},
                    "BUSINESS_LOAN": {"count": 6591},
                    "JOINT_ACCOUNT": {"count": 7098},
                }
            },
        },
    )
