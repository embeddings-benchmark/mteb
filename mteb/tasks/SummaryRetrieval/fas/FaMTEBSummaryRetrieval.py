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
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=""" """,
        descriptive_stats={"n_samples": None, "avg_character_length": None},
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
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=""" """,
        descriptive_stats={"n_samples": None, "avg_character_length": None},
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
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=""" """,
        descriptive_stats={"n_samples": None, "avg_character_length": None},
    )