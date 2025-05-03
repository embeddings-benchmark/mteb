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

        corpus[split] = split_dataset.map(
            lambda x, idx: {
                "id": f"corpus-{split}-{idx}",
                # "text": None,
                "modality": "text",
                "image": None,
            },
            with_indices=True,
            remove_columns=[
                "file_name_index",
                "class",
                "super_class",
                "sub_class",
                "split",
            ],
        )

        queries[split] = split_dataset.map(
            lambda x, idx: {
                "id": f"query-{split}-{idx}",
                "text": None,
                "modality": "image",
                # "image": None,
            },
            with_indices=True,
            remove_columns=[
                "file_name_index",
                "class",
                "super_class",
                "sub_class",
                "split",
            ],
        )
        relevant_docs[split] = {}
        for index in range(len(split_dataset)):
            query_id = f"query-{split}-{index}"
            doc_id = f"corpus-{split}-{index}"
            if query_id not in relevant_docs[split]:
                relevant_docs[split][query_id] = {}
            relevant_docs[split][query_id][doc_id] = 1
    return corpus, queries, relevant_docs


class SciMMIRI2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="SciMMIRI2TRetrieval",
        description="Retrieve captions based on figures and tables.",
        reference="https://aclanthology.org/2024.findings-acl.746/",
        dataset={
            "path": "m-a-p/SciMMIR",
            "revision": "eea276dc58c52eab33e9476acb137ff5530b78e9",
            # "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{wu2024scimmir,
  author = {Wu, Siwei and Li, Yizhi and Zhu, Kang and Zhang, Ge and Liang, Yiming and Ma, Kaijing and Xiao, Chenghao and Zhang, Haoran and Yang, Bohao and Chen, Wenhu and others},
  journal = {arXiv preprint arXiv:2401.13478},
  title = {SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval},
  year = {2024},
}
""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 261.1932607759946,
                    "average_query_length": 1.0,
                    "num_documents": 16263,
                    "num_queries": 16263,
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
