from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


def _load_data(
    path: str,
    splits: str,
    cache_dir: str | None = None,
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
        queries[split] = query_ds

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
        corpus[split] = corpus_ds

        qrels_ds = load_dataset(
            path,
            "qrels",
            split=split,
            cache_dir=cache_dir,
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


class VidoreArxivQARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VidoreArxivQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/arxivqa_test_subsampled_beir",
            "revision": "7d94d570960eac2408d3baa7a33f9de4822ae3e4",
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
        bibtex_citation="""@article{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 99.328,
                    "num_documents": 500,
                    "num_queries": 500,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class VidoreDocVQARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VidoreDocVQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/docvqa_test_subsampled_beir",
            "revision": "162ba2fc1a8437eda8b6c37b240bc1c0f0deb092",
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
        bibtex_citation="""@article{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 41.746,
                    "num_documents": 500,
                    "num_queries": 500,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class VidoreInfoVQARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VidoreInfoVQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/infovqa_test_subsampled_beir",
            "revision": "b802cc5fd6c605df2d673a963667d74881d2c9a4",
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
        bibtex_citation="""@article{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 64.934,
                    "num_documents": 500,
                    "num_queries": 500,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class VidoreTabfquadRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VidoreTabfquadRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/tabfquad_test_subsampled_beir",
            "revision": "61a2224bcd29b7b261a4892ff4c8bea353527a31",
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
        bibtex_citation="""@article{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 100.63214285714285,
                    "num_documents": 70,
                    "num_queries": 280,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class VidoreTatdqaRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VidoreTatdqaRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/tatdqa_test_beir",
            "revision": "5feb5630fdff4d8d189ffedb2dba56862fdd45c0",
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
        bibtex_citation="""@article{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 72.76368009621167,
                    "num_documents": 277,
                    "num_queries": 1663,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class VidoreShiftProjectRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VidoreShiftProjectRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/shiftproject_test_beir",
            "revision": "84a382e05c4473fed9cff2bbae95fe2379416117",
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
        bibtex_citation="""@article{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 97.7,
                    "num_documents": 1000,
                    "num_queries": 100,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class VidoreSyntheticDocQAAIRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAAIRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/syntheticDocQA_artificial_intelligence_test_beir",
            "revision": "2d9ebea5a1c6e9ef4a3b902a612f605dca11261c",
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
        bibtex_citation="""@article{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 77.71,
                    "num_documents": 968,
                    "num_queries": 100,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class VidoreSyntheticDocQAEnergyRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAEnergyRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/syntheticDocQA_energy_test_beir",
            "revision": "9935aadbad5c8deec30910489db1b2c7133ae7a7",
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
        bibtex_citation="""@article{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 83.69,
                    "num_documents": 977,
                    "num_queries": 100,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class VidoreSyntheticDocQAGovernmentReportsRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAGovernmentReportsRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/syntheticDocQA_government_reports_test_beir",
            "revision": "b4909afa930f81282fd20601e860668073ad02aa",
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
        bibtex_citation="""@article{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 82.53,
                    "num_documents": 972,
                    "num_queries": 100,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True


class VidoreSyntheticDocQAHealthcareIndustryRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VidoreSyntheticDocQAHealthcareIndustryRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/syntheticDocQA_healthcare_industry_test_beir",
            "revision": "f9e25d5b6e13e1ad9f5c3cce202565031b3ab164",
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
        bibtex_citation="""@article{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}""",
        prompt={"query": "Find a screenshot that relevant to the user's question."},
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 80.43,
                    "num_documents": 965,
                    "num_queries": 100,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
