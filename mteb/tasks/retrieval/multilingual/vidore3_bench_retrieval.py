from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGS = {
    "french": ["fra-Latn"],
    "spanish": ["spa-Latn"],
    "english": ["eng-Latn"],
    "german": ["deu-Latn"],
    "italian": ["ita-Latn"],
    "portuguese": ["por-Latn"],
}


def _load_data(
    path: str,
    splits: list[str],
    langs: list | None = None,
    revision: str | None = None,
):
    query_columns_to_remove = [
        "query_id",
        "query",
        "query_types",
        "query_format",
        "content_type",
        "written_answers",
        "query_generator",
        "query_generation_pipeline",
        "source_type",
        "query_type_for_generation",
        "gold_answer",
    ]

    qrel_columns_to_remove = ["content_type", "bounding_boxes"]

    corpus_columns_to_remove = [
        "corpus_id",
        "doc_id",
        "markdown",
        "page_number_in_doc",
    ]

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
            revision=revision,
        )
        query_ds = query_ds.map(
            lambda x: {
                "id": f"query-{split}-{x['query_id']}",
                "text": x["query"],
            },
            remove_columns=query_columns_to_remove,
        )

        corpus_ds = load_dataset(
            path,
            "corpus",
            split=split,
            revision=revision,
        )
        corpus_ds = corpus_ds.map(
            lambda x: {
                "id": f"corpus-{split}-{x['corpus_id']}",
            },
            remove_columns=corpus_columns_to_remove,
        )

        qrels_ds = load_dataset(
            path,
            "qrels",
            split=split,
            revision=revision,
        )
        qrels_ds = qrels_ds.remove_columns(qrel_columns_to_remove)

        if langs is None:
            queries[split] = query_ds
            corpus[split] = corpus_ds
            relevant_docs[split] = {}
            for row in qrels_ds:
                qid = f"query-{split}-{row['query_id']}"
                did = f"corpus-{split}-{row['corpus_id']}"
                if qid not in relevant_docs[split]:
                    relevant_docs[split][qid] = {}
                relevant_docs[split][qid][did] = int(row["score"])
        else:
            for lang in langs:
                queries[lang][split] = query_ds.filter(lambda x: x["language"] == lang)

                corpus[lang][split] = corpus_ds

                relevant_docs[lang][split] = {}
                for row in qrels_ds:
                    qid = f"query-{split}-{row['query_id']}"
                    did = f"corpus-{split}-{row['corpus_id']}"
                    if qid not in relevant_docs[lang][split]:
                        relevant_docs[lang][split][qid] = {}
                    relevant_docs[lang][split][qid][did] = int(row["score"])

    return corpus, queries, relevant_docs


class Vidore3FinanceBankReportsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceBankReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mysecretorga/finance_bank_reports_en",
            "revision": "test",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
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
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            # revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class Vidore3LuxuryFinancialReportsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3LuxuryFinancialReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mysecretorga/luxury_financial_reports_fr",
            "revision": "198cc9f5d95cdb0e4ca20fbbae4af183ed0ab824",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}""
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            # revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class Vidore3MilitaryTechnicalReportsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3MilitaryTechnicalReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mysecretorga/military_technical_reports",
            "revision": "test",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
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
}""
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            # revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class Vidore3HealthcareFdaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3HealthcareFdaRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mysecretorga/healthcare_fda_en",
            "revision": "test",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
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
}""
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            # revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class Vidore3EducationComputerScienceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3EducationComputerScienceRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mysecretorga/education_computer_science_en",
            "revision": "test",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
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
}""
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            # revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class Vidore3HrEuropeanReportsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3HrEuropeanReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mysecretorga/hr_european_reports_en",
            "revision": "test",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
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
}""
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            # revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class Vidore3EnergyReportsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3EnergyReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mysecretorga/energy_reports_fr",
            "revision": "test",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
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
}""
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            # revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class Vidore3EducationPhysicsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3EducationPhysicsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mysecretorga/education_physics_fr",
            "revision": "test",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
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
}""
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            # revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class Vidore3EnergyNuclearRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3EnergyNuclearRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mysecretorga/energy_nuclear_en",
            "revision": "test",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
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
}""
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=False,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            # revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class Vidore3TelecomTechnicalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3TelecomTechnicalRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mysecretorga/telecom_technical_en",
            "revision": "test",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
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
}""
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=False,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            # revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
