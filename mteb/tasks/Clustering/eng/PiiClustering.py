from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class PiiClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="PiiClustering",
        description="Synthetic financial documents containing Personally Identifiable Information (PII)",
        reference="https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual",
        dataset={
            "path": "FinanceMTEB/synthetic_pii_finance_en",
            "revision": "5021671d60a324d576a7b57e4c4e13bfcf857a4d",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2024-06-01", "2024-06-01"),
        domains=["Finance"],
        license="apache-2.0",
        annotations_creators="LM-generated",
        bibtex_citation="""@software{gretel-synthetic-pii-finance-multilingual-2024,
          author = {Watson, Alex and Meyer, Yev and Van Segbroeck, Maarten and Grossman, Matthew and Torbey, Sami and Mlocek, Piotr and Greco, Johnny},
          title = {{Synthetic-PII-Financial-Documents-North-America}: A synthetic dataset for training language models to label and detect PII in domain specific formats},
          month = {June},
          year = {2024},
          url = {https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual}
        }
""",
        descriptive_stats={
            "num_samples": {"test": 32},
            "average_text_length": {"test": 3425.5},
            "average_labels_per_text": {"test": 3425.5},
            "unique_labels": {"test": 29},
            "labels": {
                "test": {
                    "Corporate Tax Return": {"count": 5184},
                    "Compliance Certificate": {"count": 1920},
                    "Customer support conversational log": {"count": 6080},
                    "Currency Exchange Rate Sheet": {"count": 2008},
                    "Bank Statement": {"count": 2048},
                    "Real Estate Loan Agreement": {"count": 1984},
                    "Payment Confirmation": {"count": 4080},
                    "EDI": {"count": 6352},
                    "Financial Aid Application": {"count": 1944},
                    "Financial Forecast": {"count": 1944},
                    "Financial Regulatory Compliance Report": {"count": 1984},
                    "Loan Agreement": {"count": 2200},
                    "Credit Card Application": {"count": 2216},
                    "Business Plan": {"count": 2064},
                    "Financial Data Feed": {"count": 2080},
                    "Transaction Confirmation": {"count": 2104},
                    "ISDA Definition": {"count": 6336},
                    "FIX Protocol": {"count": 6904},
                    "Policyholder's Report": {"count": 5784},
                    "Insurance Claim Form": {"count": 2200},
                    "Dispute Resolution Policy": {"count": 2112},
                    "Tax Return": {"count": 4528},
                    "FpML": {"count": 12736},
                    "CSV": {"count": 6544},
                    "Mortgage Amortization Schedule": {"count": 2352},
                    "Health Insurance Claim Form": {"count": 3880},
                    "Mortgage Contract": {"count": 2008},
                    "Financial Disclosure Statement": {"count": 2136},
                    "BAI Format": {"count": 5904},
                }
            },
        },
    )
