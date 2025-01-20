from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SynPerQARetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="SynPerQARetrieval",
        description="Synthetic Persian QA Retrieval",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-qa-retrieval/settings",
        dataset={
            "path": "MCINext/synthetic-persian-qa-retrieval",
            "revision": "e85114f13f42dc1edc456d58931cc38d44d697cf",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation="""""",
    )

class SynPerChatbotTopicsRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="SynPerChatbotTopicsRetrieval",
        description="Synthetic Persian Chatbot Topics Retrieval",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-topics-retrieval",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-topics-retrieval",
            "revision": "576cb1b7b1ba5fd37501809964c7d7ab623fc9cb",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation="""""",
    )

class SynPerChatbotRAGTopicsRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="SynPerChatbotRAGTopicsRetrieval",
        description="Synthetic Persian Chatbot RAG Topics Retrieval",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-rag-topics-retrieval",
            "revision": "66a4a07f13ec255e12a9f071e588dfa269459017",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation="""""",
    )


class SynPerChatbotRAGFAQRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="SynPerChatbotRAGFAQRetrieval",
        description="Synthetic Persian Chatbot RAG FAQ Retrieval",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-faq-retrieval",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-rag-faq-retrieval",
            "revision": "861cb8fa13b31c5106ca05fedacde2fedce1284c",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation="""""",
    )


class PersianWebDocumentRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="PersianWebDocumentRetrieval",
        description="Persian dataset designed specifically for the task of text information retrieval through the web.",
        reference="https://ieeexplore.ieee.org/document/10553090",
        dataset={
            "path": "MCINext/persian-web-document-retrieval",
            "revision": "2f0e257943c5772ce9eaef8988581408ec0beb95",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""""",
    )