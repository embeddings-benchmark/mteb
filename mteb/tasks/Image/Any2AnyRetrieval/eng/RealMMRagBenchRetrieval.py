from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _load_data(
    path: str,
    splits: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    text_col: str = "query",
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
        description="""Contains annual financial reports rich in text, tables, and figures from IBM’s public filings.
                    Queries ask about financial results, trends, or statements across multiple years.
                    Retrieval goal: find the specific report page containing the relevant financial information.""",
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
  author = {Wasserman, Navve and Pony, Roi and Naparstek, Oshri and Goldfarb, Adi Raz and Schwartz, Eli and Barzelay, Udi and Karlinsky, Leonid},
  journal = {arXiv preprint arXiv:2502.12342},
  title = {REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
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
        description="""Comprises quarterly investor presentation slides combining tables, charts, and textual highlights.
                    Queries focus on revenue trends, growth metrics, or business segments.
                    Retrieval goal: retrieve the slide that presents the requested financial data or insight.""",
        reference="https://arxiv.org/abs/2502.12342",
        dataset={
            "path": "ibm-research/REAL-MM-RAG_FinSlides_BEIR",
            "revision": "41167605aed3ab0ff342ac8f318163c6e59b8b31",
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
  author = {Wasserman, Navve and Pony, Roi and Naparstek, Oshri and Goldfarb, Adi Raz and Schwartz, Eli and Barzelay, Udi and Karlinsky, Leonid},
  journal = {arXiv preprint arXiv:2502.12342},
  title = {REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
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
        description="""Includes technical documentation and whitepapers on IBM storage and automation systems with text-heavy content and supporting visuals.
                    Queries address specific technologies, architectures, or performance aspects.
                    Retrieval goal: locate the report page providing the technical explanation or result.""",
        reference="https://arxiv.org/abs/2502.12342",
        dataset={
            "path": "ibm-research/REAL-MM-RAG_TechReport_BEIR",
            "revision": "13642f1f8d39e032757f4d0ee73814452fc76d17",
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
  author = {Wasserman, Navve and Pony, Roi and Naparstek, Oshri and Goldfarb, Adi Raz and Schwartz, Eli and Barzelay, Udi and Karlinsky, Leonid},
  journal = {arXiv preprint arXiv:2502.12342},
  title = {REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
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
        description="""Features technical presentation slides containing bullet points, flow diagrams, and schematic figures.
                    Queries reflect realistic information-seeking about system design or AI and automation concepts.
                    Retrieval goal: retrieve the slide that best answers the technical query through text and visuals.""",
        reference="https://arxiv.org/abs/2502.12342",
        dataset={
            "path": "ibm-research/REAL-MM-RAG_TechSlides_BEIR",
            "revision": "614ad5cac2edd86756045f04075d335a3825a692",
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
  author = {Wasserman, Navve and Pony, Roi and Naparstek, Oshri and Goldfarb, Adi Raz and Schwartz, Eli and Barzelay, Udi and Karlinsky, Leonid},
  journal = {arXiv preprint arXiv:2502.12342},
  title = {REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
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
