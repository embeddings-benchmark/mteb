from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class ComplaintsClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="ComplaintsClustering",
        description="The Consumer Complaint Database is a collection of complaints about consumer financial products and services that sent to companies for response..",
        dataset={
            "path": "FinanceMTEB/Complaints",
            "revision": "6704122294b7693f5e544cdde1e4a3e80b291b76",
        },
        reference="https://huggingface.co/datasets/CFPB/consumer-finance-complaints",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2022-01-01", "2022-12-31"),
        domains=["Finance"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        bibtex_citation="""@misc{CFPB-consumer-finance-complaints,
        title={Consumer Finance Complaints},
        author={Consumer Financial Protection Bureau},
        howpublished={\\url{https://huggingface.co/datasets/CFPB/consumer-finance-complaints}},
        year={2022},
        note={Accessed: [DATE]}
        }""",
        descriptive_stats={
            "num_samples": {"test": 16},
            "average_text_length": {"test": 35624.5},
            "average_labels_per_text": {"test": 35624.5},
            "unique_labels": {"test": 12},
            "labels": {
                "test": {
                    "Credit reporting, credit repair services, or other personal consumer reports": {
                        "count": 40000
                    },
                    "Debt collection": {"count": 80000},
                    "Payday loan": {"count": 27936},
                    "Credit card": {"count": 40000},
                    "Credit reporting": {"count": 80000},
                    "Vehicle loan or lease": {"count": 40000},
                    "Prepaid card": {"count": 34896},
                    "Student loan": {"count": 80000},
                    "Checking or savings account": {"count": 40000},
                    "Consumer Loan": {"count": 40000},
                    "Payday loan, title loan, or personal loan": {
                        "count": 40000
                    },
                    "Payday loan, title loan, personal loan, or advance loan": {
                        "count": 27160
                    },
                }
            },
        },
    )
