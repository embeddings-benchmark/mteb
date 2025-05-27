from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = {
    "french": ["fra-Latn"],
    "spanish": ["spa-Latn"],
    "english": ["eng-Latn"],
    "german": ["deu-Latn"],
}


def _load_data(
    path: str,
    splits: str,
    langs: list | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    if langs is None:
        corpus = {}
        queries = {}
        relevant_docs = {}
    else:
        corpus = {lang: {} for lang in langs}
        queries = {lang: {} for lang in langs}
        relevant_docs = {lang: {} for lang in langs}

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
                "text": x["query"],
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

        if langs is None:
            queries[split] = query_ds
            corpus[split] = corpus_ds
            relevant_docs[split] = {}
            for row in qrels_ds:
                qid = f"query-{split}-{row['query-id']}"
                did = f"corpus-{split}-{row['corpus-id']}"
                if qid not in relevant_docs[split]:
                    relevant_docs[split][qid] = {}
                relevant_docs[split][qid][did] = int(row["score"])
        else:
            for lang in langs:
                queries[lang][split] = query_ds.filter(lambda x: x["language"] == lang)

                corpus[lang][split] = corpus_ds

                relevant_docs[lang][split] = {}
                for row in qrels_ds:
                    qid = f"query-{split}-{row['query-id']}"
                    did = f"corpus-{split}-{row['corpus-id']}"
                    if qid not in relevant_docs[lang][split]:
                        relevant_docs[lang][split][qid] = {}
                    relevant_docs[lang][split][qid][did] = int(row["score"])

    return corpus, queries, relevant_docs


class Vidore2ESGReportsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Vidore2ESGReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/esg_reports_v2",
            "revision": "0542c0d03da0ec1c8cbc517c8d78e7e95c75d3d3",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 30,
                    "num_queries": 228,
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
            langs=_LANGS.keys(),
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class Vidore2EconomicsReportsRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Vidore2EconomicsReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/economics_reports_v2",
            "revision": "b3e3a04b07fbbaffe79be49dabf92f691fbca252",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 5,
                    "num_queries": 232,
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
            langs=_LANGS.keys(),
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class Vidore2BioMedicalLecturesRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Vidore2BioMedicalLecturesRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/biomedical_lectures_v2",
            "revision": "a29202f0da409034d651614d87cd8938d254e2ea",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 27,
                    "num_queries": 640,
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
            langs=_LANGS.keys(),
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class Vidore2ESGReportsHLRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Vidore2ESGReportsHLRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/esg_reports_human_labeled_v2",
            "revision": "6d467dedb09a75144ede1421747e47cf036857dd",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2025-01-01", "2025-03-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "num_documents": 27,
                    "num_queries": 640,
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
