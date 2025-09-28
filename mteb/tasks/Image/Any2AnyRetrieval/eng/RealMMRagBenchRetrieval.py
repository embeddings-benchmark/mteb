from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


def _load_data(
    path: str,
    splits: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    text_col: str | "query" = "query",
):
    corpus = {}
    queries = {}
    relevant_docs = {}


    for split in splits:
        query_ds = load_dataset(
            path,
            "queries",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        query_ds = query_ds.map(
            lambda x: {
                "id": f"query-{split}-{x['query-id']}",
                "text": x[text_col],
                "image": None,
                "modality": "text",
            },
            remove_columns=["query-id", "query"],
        )

        corpus_ds = load_dataset(
            path,
            "corpus",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        corpus_ds = corpus_ds.map(
            lambda x: {
                "id": f"corpus-{split}-{x['corpus-id']}",
                "text": None,
                "modality": "image",
            },
            remove_columns=["corpus-id"],
        )

        qrels_ds = load_dataset(
            path,
            "qrels",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )

        queries[split] = query_ds
        corpus[split] = corpus_ds
        relevant_docs[split] = {}
        for row in qrels_ds:
            qid = f"query-{split}-{row['query-id']}"
            did = f"corpus-{split}-{row['corpus-id']}"
            if qid not in relevant_docs[split]:
                relevant_docs[split][qid] = {}
            relevant_docs[split][qid][did] = int(row["score"])

    return corpus, queries, relevant_docs


class RealMMRagFinReportRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="RealMMRagFinReportRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/abs/2502.12342",
        dataset={
            "path": "ibm-research/REAL-MM-RAG_FinReport_BEIR",
            "revision": "e66ef8cc883d823483db7b5b71065eb7c1dae12c",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cdla-sharing-1.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{wasserman2025real,
  title={REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark},
  author={Wasserman, Navve and Pony, Roi and Naparstek, Oshri and Goldfarb, Adi Raz and Schwartz, Eli and Barzelay, Udi and Karlinsky, Leonid},
  journal={arXiv preprint arXiv:2502.12342},
  year={2025}
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 141.5,
                    "num_documents": 19,
                    "num_queries": 853,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True

class RealMMRagFinSlidesRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="RealMMRagFinSlidesRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/abs/2502.12342",
        dataset={
            "path": "ibm-research/REAL-MM-RAG_FinSlides_BEIR",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cdla-sharing-1.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{wasserman2025real,
  title={REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark},
  author={Wasserman, Navve and Pony, Roi and Naparstek, Oshri and Goldfarb, Adi Raz and Schwartz, Eli and Barzelay, Udi and Karlinsky, Leonid},
  journal={arXiv preprint arXiv:2502.12342},
  year={2025}
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 35,
                    "num_documents": 65,
                    "num_queries": 1052,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        self.data_loaded = True


class RealMMRagTechReportRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="RealMMRagTechReportRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/abs/2502.12342",
        dataset={
            "path": "ibm-research/REAL-MM-RAG_TechReport_BEIR",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cdla-sharing-1.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{wasserman2025real,
  title={REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark},
  author={Wasserman, Navve and Pony, Roi and Naparstek, Oshri and Goldfarb, Adi Raz and Schwartz, Eli and Barzelay, Udi and Karlinsky, Leonid},
  journal={arXiv preprint arXiv:2502.12342},
  year={2025}
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 98.5,
                    "num_documents": 17,
                    "num_queries": 1294,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        self.data_loaded = True


class RealMMRagTechSlidesRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="RealMMRagTechSlidesRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/abs/2502.12342",
        dataset={
            "path": "ibm-research/REAL-MM-RAG_TechSlides_BEIR",
            "revision": "main",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cdla-sharing-1.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{wasserman2025real,
  title={REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark},
  author={Wasserman, Navve and Pony, Roi and Naparstek, Oshri and Goldfarb, Adi Raz and Schwartz, Eli and Barzelay, Udi and Karlinsky, Leonid},
  journal={arXiv preprint arXiv:2502.12342},
  year={2025}
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 31.7,
                    "num_documents": 62,
                    "num_queries": 1354,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        self.data_loaded = True


