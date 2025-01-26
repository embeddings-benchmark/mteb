from __future__ import annotations

from mteb.abstasks.AbsTaskSummaryRetrieval import AbsTaskSummaryRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SAMSumFa(AbsTaskSummaryRetrieval):
    metadata = TaskMetadata(
        name="SAMSumFa",
        description="Translated Version of SAMSum Dataset",
        reference="https://huggingface.co/datasets/MCINext/samsum-fa",
        dataset={
            "path": "MCINext/samsum-fa",
            "revision": "fd981d78a0ab82c20d2e693a8b3929c5d71b0743",
        },
        type="SummaryRetrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="f1",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=""" """,
        descriptive_stats={
            "test": {
                "num_samples": 1561,
                "number_of_characters": 930587,
                "unique_pairs": 1561,
                "min_text_length": 46,
                "average_text_length": 489.58872517616913,
                "max_text_length": 2802,
                "unique_text": 1561,
                "min_summary_length": 14,
                "average_summary_length": 106.55925688661115,
                "max_summary_length": 325,
                "unique_summary": 1561,
            }
        },
    )


class SynPerChatbotSumSRetrieval(AbsTaskSummaryRetrieval):
    metadata = TaskMetadata(
        name="SynPerChatbotSumSRetrieval",
        description="Synthetic Persian Chatbot  Summary Dataset",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-summary-retrieval",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-summary-retrieval",
            "revision": "9002f5e9de4ef61f1f5c34831d2a5ed855bac0ae",
        },
        type="SummaryRetrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="f1",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
        descriptive_stats={
            "test": {
                "num_samples": 1537,
                "number_of_characters": 1701259,
                "unique_pairs": 1537,
                "min_text_length": 383,
                "average_text_length": 949.729342875732,
                "max_text_length": 1828,
                "unique_text": 1537,
                "min_summary_length": 68,
                "average_summary_length": 157.1405335068315,
                "max_summary_length": 308,
                "unique_summary": 1537,
            }
        },
    )


class SynPerChatbotRAGSumSRetrieval(AbsTaskSummaryRetrieval):
    metadata = TaskMetadata(
        name="SynPerChatbotRAGSumSRetrieval",
        description="Synthetic Persian Chatbot RAG Summary Dataset",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-summary-retrieval",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-rag-summary-retrieval",
            "revision": "f77746f286bbf2177ee7b5a803da8be440d5d4c1",
        },
        type="SummaryRetrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="f1",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=""" """,
        descriptive_stats={
            "test": {
                "num_samples": 1087,
                "number_of_characters": 835245,
                "unique_pairs": 1087,
                "min_text_length": 37,
                "average_text_length": 628.5234590616375,
                "max_text_length": 2601,
                "unique_text": 1087,
                "min_summary_length": 43,
                "average_summary_length": 139.87120515179393,
                "max_summary_length": 284,
                "unique_summary": 1087,
            }
        },
    )
