from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


def _load_data(path: str, splits: str, cache_dir: str = None, revision: str = None):
    corpus = {}
    queries = {}
    relevant_docs = {}

    dataset = load_dataset(
        path,
        cache_dir=cache_dir,
        revision=revision,
    )

    for split in splits:
        split_dataset = dataset[split]
        split_dataset = split_dataset.rename_column("query", "text")
        corpus[split] = split_dataset.map(
            lambda x, idx: {
                "id": f"corpus-{split}-{idx}",
                "modality": "image",
                "text": None,
            },
            with_indices=True,
        )

        queries[split] = split_dataset.map(
            lambda x, idx: {
                "id": f"query-{split}-{idx}",
                "image": None,
                "modality": "text",
            },
            with_indices=True,
        )
        relevant_docs[split] = {}
        for index in range(len(split_dataset)):
            query_id = f"query-{split}-{index}"
            doc_id = f"corpus-{split}-{index}"
            if query_id not in relevant_docs[split]:
                relevant_docs[split][query_id] = {}
            relevant_docs[split][query_id][doc_id] = 1
    return corpus, queries, relevant_docs


def _load_data_qc_unmatched(
    path: str, splits: str, cache_dir: str = None, revision: str = None, num_queries=100
):
    corpus = {}
    queries = {}
    relevant_docs = {}

    dataset = load_dataset(
        path,
        cache_dir=cache_dir,
        revision=revision,
    )

    for split in splits:
        split_dataset = dataset[split]
        split_dataset = split_dataset.rename_column("query", "text")
        corpus[split] = split_dataset.map(
            lambda x, idx: {
                "id": f"corpus-{split}-{idx}",
                "modality": "image",
                "text": None,
            },
            with_indices=True,
        )

        split_dataset = split_dataset.select(range(num_queries))
        queries[split] = split_dataset.map(
            lambda x, idx: {
                "id": f"query-{split}-{idx}",
                "image": None,
                "modality": "text",
            },
            with_indices=True,
        )
        relevant_docs[split] = {}
        for index in range(len(queries[split])):
            query_id = f"query-{split}-{index}"
            doc_id = f"corpus-{split}-{index}"
            if query_id not in relevant_docs[split]:
                relevant_docs[split][query_id] = {}
            relevant_docs[split][query_id][doc_id] = 1
    return corpus, queries, relevant_docs


class VidoreArxivQARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VidoreArxivQARetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/pdf/2407.01449",
        dataset={
            "path": "vidore/arxivqa_test_subsampled",
            "revision": "fe2b0e055eaac82d8f6801ebc8e85d8832248133",
        },
        type="Any2AnyRetrieval",
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
            "path": "vidore/docvqa_test_subsampled",
            "revision": "b1d89eda849e636676df6ead8002602fb1858600",
        },
        type="Any2AnyRetrieval",
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
            "path": "vidore/infovqa_test_subsampled",
            "revision": "fec9c59496ddf4a34e01ca8080515722bd3cf970",
        },
        type="Any2AnyRetrieval",
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
            "path": "vidore/tabfquad_test_subsampled",
            "revision": "501f02a80aff50c90045b0feaa81565c4e8f889e",
        },
        type="Any2AnyRetrieval",
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
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 100.63214285714285,
                    "num_documents": 280,
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
            "path": "vidore/tatdqa_test",
            "revision": "9c3a626c16c811f15514689c3e7e95a4f2b9b8c3",
        },
        type="Any2AnyRetrieval",
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
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 72.76368009621167,
                    "num_documents": 1663,
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
            "path": "vidore/shiftproject_test",
            "revision": "9e7df4c35994683a7ba88002fb22917ffa15067e",
        },
        type="Any2AnyRetrieval",
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
        self.corpus, self.queries, self.relevant_docs = _load_data_qc_unmatched(
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
            "path": "vidore/syntheticDocQA_artificial_intelligence_test",
            "revision": "5fe59d7e52732b86d11ee0e9c4a8cdb0e8ba7a6e",
        },
        type="Any2AnyRetrieval",
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
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 77.71,
                    "num_documents": 1000,
                    "num_queries": 100,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data_qc_unmatched(
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
            "path": "vidore/syntheticDocQA_energy_test",
            "revision": "0821bc71310cfa51d5c8131d4d8b9c4d537bd8c8",
        },
        type="Any2AnyRetrieval",
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
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 83.69,
                    "num_documents": 1000,
                    "num_queries": 100,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data_qc_unmatched(
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
            "path": "vidore/syntheticDocQA_government_reports_test",
            "revision": "8270b3751ce6b95bec362fb38fbcd2a4aa400cfc",
        },
        type="Any2AnyRetrieval",
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
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 82.53,
                    "num_documents": 1000,
                    "num_queries": 100,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data_qc_unmatched(
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
            "path": "vidore/syntheticDocQA_healthcare_industry_test",
            "revision": "86f09ebc1703516c76e5f931465e2ed7626a5e52",
        },
        type="Any2AnyRetrieval",
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
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 80.43,
                    "num_documents": 1000,
                    "num_queries": 100,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data_qc_unmatched(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
