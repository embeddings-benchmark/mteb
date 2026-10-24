from typing import Any

from datasets import Features, Value, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX_CITATION = r"""
@inproceedings{wasserman-etal-2025-real,
  address = {Vienna, Austria},
  author = {Wasserman, Navve and Pony, Roi and Naparstek, Oshri and Goldfarb, Adi Raz and Schwartz, Eli and Barzelay, Udi and Karlinsky, Leonid},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2025.acl-long.1528},
  editor = {Che, Wanxiang and Nabende, Joyce and Shutova, Ekaterina and Pilehvar, Mohammad Taher},
  isbn = {979-8-89176-251-0},
  month = jul,
  pages = {31660--31683},
  publisher = {Association for Computational Linguistics},
  title = {{REAL}-{MM}-{RAG}: A Real-World Multi-Modal Retrieval Benchmark},
  url = {https://aclanthology.org/2025.acl-long.1528/},
  year = {2025},
}
"""


def _load_data(
    path: str,
    revision: str,
    splits: list[str],
    num_proc: int | None,
) -> dict[str, dict[str, RetrievalSplitData]]:
    dataset: dict[str, dict[str, RetrievalSplitData]] = {"default": {}}

    for split in splits:
        queries = load_dataset(
            path,
            "queries",
            split=split,
            revision=revision,
            num_proc=num_proc,
        )
        queries = (
            queries.cast_column("query-id", Value("string"))
            .rename_column("query-id", "id")
            .rename_column("rephrase_level_3", "text")
            .select_columns(["id", "text"])
        )

        corpus = load_dataset(
            path,
            "corpus",
            split=split,
            revision=revision,
            num_proc=num_proc,
        )
        corpus = (
            corpus.cast_column("corpus-id", Value("string"))
            .rename_column("corpus-id", "id")
            .select_columns(["id", "image"])
        )

        qrels = load_dataset(
            path,
            "qrels",
            split=split,
            revision=revision,
            num_proc=num_proc,
        )
        qrels = qrels.select_columns(["query-id", "corpus-id", "score"]).cast(
            Features(
                {
                    "query-id": Value("string"),
                    "corpus-id": Value("string"),
                    "score": Value("int32"),
                }
            )
        )
        qrels = qrels.to_polars()
        relevant_docs = {
            query_id[0]: dict(zip(group["corpus-id"], group["score"], strict=True))
            for query_id, group in qrels.group_by("query-id", maintain_order=False)
        }

        dataset["default"][split] = RetrievalSplitData(
            corpus=corpus,
            queries=queries,
            relevant_docs=relevant_docs,
            top_ranked=None,
        )

    return dataset


class RealMMRAGFinReportRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RealMMRAGFinReportRetrieval",
        description="REAL-MM-RAG FinReport retrieves relevant page images from 19 financial reports containing text and tables. Evaluation uses the benchmark's most strongly rephrased (level 3) queries.",
        reference="https://aclanthology.org/2025.acl-long.1528/",
        dataset={
            "path": "ibm-research/REAL-MM-RAG_FinReport_BEIR",
            "revision": "e66ef8cc883d823483db7b5b71065eb7c1dae12c",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2005-01-01", "2023-12-31"),
        domains=["Non-fiction"],
        task_subtypes=["Image Text Retrieval"],
        license="https://cdla.dev/permissive-2-0/",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        self.dataset = _load_data(
            path=self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            splits=self.metadata.eval_splits,
            num_proc=num_proc,
        )
        self.data_loaded = True


class RealMMRAGFinSlidesRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RealMMRAGFinSlidesRetrieval",
        description="REAL-MM-RAG FinSlides retrieves relevant page images from 65 table-heavy quarterly financial presentations. Evaluation uses the benchmark's most strongly rephrased (level 3) queries.",
        reference="https://aclanthology.org/2025.acl-long.1528/",
        dataset={
            "path": "ibm-research/REAL-MM-RAG_FinSlides_BEIR",
            "revision": "41167605aed3ab0ff342ac8f318163c6e59b8b31",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2008-01-01", "2024-12-31"),
        domains=["Non-fiction"],
        task_subtypes=["Image Text Retrieval"],
        license="https://cdla.dev/permissive-2-0/",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        self.dataset = _load_data(
            path=self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            splits=self.metadata.eval_splits,
            num_proc=num_proc,
        )
        self.data_loaded = True


class RealMMRAGTechReportRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RealMMRAGTechReportRetrieval",
        description="REAL-MM-RAG TechReport retrieves relevant page images from 17 text-heavy IBM FlashSystem documents containing visual elements and tables. Evaluation uses the benchmark's most strongly rephrased (level 3) queries.",
        reference="https://aclanthology.org/2025.acl-long.1528/",
        dataset={
            "path": "ibm-research/REAL-MM-RAG_TechReport_BEIR",
            "revision": "13642f1f8d39e032757f4d0ee73814452fc76d17",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2025-02-18"),
        domains=["Non-fiction"],
        task_subtypes=["Image Text Retrieval"],
        license="https://cdla.dev/permissive-2-0/",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        self.dataset = _load_data(
            path=self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            splits=self.metadata.eval_splits,
            num_proc=num_proc,
        )
        self.data_loaded = True


class RealMMRAGTechSlidesRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RealMMRAGTechSlidesRetrieval",
        description="REAL-MM-RAG TechSlides retrieves relevant page images from 62 technical presentations on business and IT automation. Evaluation uses the benchmark's most strongly rephrased (level 3) queries.",
        reference="https://aclanthology.org/2025.acl-long.1528/",
        dataset={
            "path": "ibm-research/REAL-MM-RAG_TechSlides_BEIR",
            "revision": "614ad5cac2edd86756045f04075d335a3825a692",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2025-02-18"),
        domains=["Non-fiction"],
        task_subtypes=["Image Text Retrieval"],
        license="https://cdla.dev/permissive-2-0/",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=_BIBTEX_CITATION,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        self.dataset = _load_data(
            path=self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            splits=self.metadata.eval_splits,
            num_proc=num_proc,
        )
        self.data_loaded = True
