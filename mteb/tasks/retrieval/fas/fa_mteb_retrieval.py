from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

fa_mteb = """@inproceedings{zinvandi2025famteb,
  address = {Suzhou, China},
  author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2025},
  doi = {10.18653/v1/2025.findings-emnlp.614},
  editor = {Christodoulopoulos, Christos and Chakraborty, Tanmoy and Rose, Carolyn and Peng, Violet},
  isbn = {979-8-89176-335-7},
  month = {nov},
  pages = {11441--11468},
  publisher = {Association for Computational Linguistics},
  title = {{F}a{MTEB}: Massive Text Embedding Benchmark in {P}ersian Language},
  url = {https://aclanthology.org/2025.findings-emnlp.614/},
  year = {2025},
}"""


class SynPerQARetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="SynPerQARetrieval",
        description="Synthetic Persian QA Retrieval",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-qa-retrieval/",
        dataset={
            "path": "MCINext/synthetic-persian-qa-retrieval",
            "revision": "e85114f13f42dc1edc456d58931cc38d44d697cf",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=fa_mteb,
    )


class SynPerChatbotTopicsRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="SynPerChatbotTopicsRetrieval",
        description="Synthetic Persian Chatbot Topics Retrieval",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-topics-retrieval",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-topics-retrieval",
            "revision": "086995ca4cea33f37a407c2fa5282f74913740ee",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=fa_mteb,
    )


class SynPerChatbotRAGTopicsRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="SynPerChatbotRAGTopicsRetrieval",
        description="Synthetic Persian Chatbot RAG Topics Retrieval",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-rag-topics-retrieval",
            "revision": "da8f36a723da155738f5e3d8d84d543589bd5083",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=fa_mteb,
    )


class SynPerChatbotRAGFAQRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="SynPerChatbotRAGFAQRetrieval",
        description="Synthetic Persian Chatbot RAG FAQ Retrieval",
        reference="https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-faq-retrieval",
        dataset={
            "path": "MCINext/synthetic-persian-chatbot-rag-faq-retrieval",
            "revision": "9d32af6540970e2845028cbfffe6b0d0e8f52428",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Question answering"],
        license="not specified",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=fa_mteb,
    )


class PersianWebDocumentRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True
    metadata = TaskMetadata(
        name="PersianWebDocumentRetrieval",
        description="Persian dataset designed specifically for the task of text information retrieval through the web.",
        reference="https://ieeexplore.ieee.org/document/10553090",
        dataset={
            "path": "MCINext/persian-web-document-retrieval",
            "revision": "b3dc818368a867b30ccb55a42ff287d253512c36",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fas-Arab"],
        main_score="ndcg_at_10",
        date=("2024-09-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=fa_mteb,
    )
