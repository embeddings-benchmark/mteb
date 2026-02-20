from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _load_data(
    path: str,
    splits: str,
    revision: str | None = None,
):
    corpus = {}
    queries = {}
    relevant_docs = {}

    for split in splits:
        query_ds = load_dataset(
            path,
            "queries",
            split=split,
            revision=revision,
        )
        query_ds = query_ds.map(
            lambda x: {
                "id": f"query-{split}-{x['query-id']}",
                "text": x["query"],
                "modality": "text",
            },
            remove_columns=["query-id", "query"],
        )
        queries[split] = query_ds

        corpus_ds = load_dataset(
            path,
            "corpus",
            split=split,
            revision=revision,
        )
        corpus_ds = corpus_ds.map(
            lambda x: {
                "id": f"corpus-{split}-{x['corpus-id']}",
                "modality": "image",
            },
            remove_columns=["corpus-id"],
        )
        corpus[split] = corpus_ds

        qrels_ds = load_dataset(
            path,
            "qrels",
            split=split,
            revision=revision,
        )
        relevant_docs[split] = {}
        for row in qrels_ds:
            qid = f"query-{split}-{row['query-id']}"
            did = f"corpus-{split}-{row['corpus-id']}"
            if qid not in relevant_docs[split]:
                relevant_docs[split][qid] = {}
            relevant_docs[split][qid][did] = int(row["score"])

    return corpus, queries, relevant_docs


class VidoreArxivQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreArxivQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mteb/arxivqa_test_subsampled_beir",
            "revision": "179762404662be6bf48c30b4fb2b2ef8c6891290",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class VidoreDocVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreDocVQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mteb/docvqa_test_subsampled_beir",
            "revision": "e78cd130055fcb8d69bb0ff4115c4712a157f3d3",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class VidoreInfoVQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreInfoVQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mteb/infovqa_test_subsampled_beir",
            "revision": "74d396242cc281c021eeb38d0ee5f6f30afc1fad",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class VidoreTabfquadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreTabfquadRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mteb/tabfquad_test_subsampled_beir",
            "revision": "24a4541c4c461f5ff4f4af4e97ec820715281a06",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class VidoreTatdqaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreTatdqaRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mteb/tatdqa_test_beir",
            "revision": "dc7924ec102addd7f85c1ccfdd8625848179f615",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class VidoreShiftProjectRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreShiftProjectRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mteb/shiftproject_test_beir",
            "revision": "78e6f5ad3bbc1af50ae9c0f006f57462213269fb",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class VidoreSyntheticDocQAAIRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAAIRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mteb/syntheticDocQA_artificial_intelligence_test_beir",
            "revision": "12c5b7a81fc19d66f1844642acdb88e8ad79af16",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        adapted_from=["VidoreDocVQARetrieval"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class VidoreSyntheticDocQAEnergyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAEnergyRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mteb/syntheticDocQA_energy_test_beir",
            "revision": "eda52c9a0519252cfaacb7ae5e5da516b7d4b8cb",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        adapted_from=["VidoreDocVQARetrieval"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class VidoreSyntheticDocQAGovernmentReportsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAGovernmentReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mteb/syntheticDocQA_government_reports_test_beir",
            "revision": "ec534abea699035a18f9fd76f215277de9db9600",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        adapted_from=["VidoreDocVQARetrieval"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class VidoreSyntheticDocQAHealthcareIndustryRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAHealthcareIndustryRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "mteb/syntheticDocQA_healthcare_industry_test_beir",
            "revision": "81d53880628d641aaf7365f3b2e65f57c814ecca",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        adapted_from=["VidoreDocVQARetrieval"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
