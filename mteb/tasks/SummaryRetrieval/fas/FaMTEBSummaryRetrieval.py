from __future__ import annotations

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.TaskMetadata import TaskMetadata


class SAMSumFa(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="SAMSumFa",
        description="Translated Version of SAMSum Dataset for summary retrieval.",
        reference="https://huggingface.co/datasets/MCINext/samsum-fa",
        dataset={
            "path": "MCINext/samsum-fa",
            "revision": "fd981d78a0ab82c20d2e693a8b3929c5d71b0743",
        },
        type="BitextMining",
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
        sample_creation="machine-translated",
        bibtex_citation="",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"text": "sentence1", "summary": "sentence2"}
        )


class SynPerChatbotSumSRetrieval(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="SynPerChatbotSumSRetrieval",
        description="Synthetic Persian Chatbot Summary Dataset for summary retrieval.",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-summary-retrieval",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-summary-retrieval",
            "revision": "9002f5e9de4ef61f1f5c34831d2a5ed855bac0ae",
        },
        type="BitextMining",
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
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"text": "sentence1", "summary": "sentence2"}
        )


class SynPerChatbotRAGSumSRetrieval(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="SynPerChatbotRAGSumSRetrieval",
        description="Synthetic Persian Chatbot RAG Summary Dataset for summary retrieval.",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-summary-retrieval",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-rag-summary-retrieval",
            "revision": "f77746f286bbf2177ee7b5a803da8be440d5d4c1",
        },
        type="BitextMining",
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
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"text": "sentence1", "summary": "sentence2"}
        )
