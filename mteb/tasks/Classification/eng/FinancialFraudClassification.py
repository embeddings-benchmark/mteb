from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinancialFraudClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinancialFraudClassification",
        description="This dataset was used for research in detecting financial fraud.",
        reference="https://github.com/amitkedia007/Financial-Fraud-Detection-Using-LLMs",
        dataset={
            "path": "FinanceMTEB/FinancialFraud",
            "revision": "e569a69e058ad8504f03556cd05c36700767d193",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1999-01-01", "2019-12-31"),
        domains=["Finance"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        bibtex_citation="""@mastersthesis{kedia2023enhancing,
            author = {Kedia, Amit Shushil},
            title = {Enhancing Financial Fraud Detection: A Comparative Analysis of Large Language Models and Traditional Machine Learning and Deep Learning Approaches},
            school = {Brunel University London},
            year = {2023},
            address = {Uxbridge, Middlesex UB8 3PH, United Kingdom},
            type = {MSc Thesis},
            department = {Department of Computer Science},
            program = {MSc Data Science and Analytics}
        }""",
        descriptive_stats={
            "num_samples": {"test": 51},
            "average_text_length": {"test": 1096025.2156862745},
            "unique_labels": {"test": 2},
            "labels": {"test": {"0": {"count": 32}, "1": {"count": 19}}},
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
