from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset

    from mteb.types import RelevantDocumentsType

_LANGS = ["python", "javascript", "go", "ruby", "java", "php"]
_EVAL_SPLIT = "test"

logger = logging.getLogger(__name__)


def _load_code_search_code_retrieval(
    path: str, langs: list, splits: str, revision: str | None = None
) -> tuple[
    dict[str, dict[str, Dataset]],
    dict[str, dict[str, Dataset]],
    dict[str, dict[str, RelevantDocumentsType]],
]:
    corpus = {lang: {split: {} for split in splits} for lang in langs}
    queries = {lang: {split: {} for split in splits} for lang in langs}
    relevant_docs = {lang: {split: {} for split in splits} for lang in langs}

    split = _EVAL_SPLIT

    for lang in langs:
        qrels_data = datasets.load_dataset(
            path,
            name=f"{lang}-qrels",
            revision=revision,
        )[split]

        for row in qrels_data:
            query_id = row["query-id"]
            doc_id = row["corpus-id"]
            score = row["score"]
            if query_id not in relevant_docs[lang][split]:
                relevant_docs[lang][split][query_id] = {}
            relevant_docs[lang][split][query_id][doc_id] = score

        corpus_data = datasets.load_dataset(
            path,
            name=f"{lang}-corpus",
            revision=revision,
        )["corpus"]

        for row in corpus_data:
            doc_id = row["_id"]
            doc_title = row["title"]
            doc_text = row["text"]
            corpus[lang][split][doc_id] = {"title": doc_title, "text": doc_text}

        queries_data = datasets.load_dataset(
            path,
            name=f"{lang}-queries",
            revision=revision,
        )["queries"].filter(lambda x: x["partition"] == "test")

        for row in queries_data:
            query_id = row["_id"]
            query_text = row["text"]
            queries[lang][split][query_id] = query_text

        queries = queries  # noqa: PLW0127
        logger.info("Loaded %d %s Queries.", len(queries), split.upper())

    return corpus, queries, relevant_docs


class CodeSearchNetCCRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeSearchNetCCRetrieval",
        description="The dataset is a collection of code snippets. The task is to retrieve the most relevant code snippet for a given code snippet.",
        reference="https://aclanthology.org/2025.acl-long.1072/",
        dataset={
            "path": "CoIR-Retrieval/CodeSearchNet-ccr",
            "revision": "6e1effa2c03723c5fde48ee912b5ee08d4f211e8",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs={lang: [lang + "-Code"] for lang in _LANGS},
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{li2024coircomprehensivebenchmarkcode,
  address = {Vienna, Austria},
  author = {Li, Xiangyang and Dong, Kuicai and Lee, Yi Quan and Xia, Wei and Zhang, Hao and Dai, Xinyi and Wang, Yasheng and Tang, Ruiming},
  booktitle = {Proceedings of the 63rd Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2025.acl-long.1072},
  editor = {Che, Wanxiang and Nabende, Joyce and Shutova, Ekaterina and Pilehvar, Mohammad Taher},
  isbn = {979-8-89176-251-0},
  month = {jul},
  pages = {22074--22091},
  publisher = {Association for Computational Linguistics},
  title = {{C}o{IR}: A Comprehensive Benchmark for Code Information Retrieval Models},
  url = {https://aclanthology.org/2025.acl-long.1072/},
  year = {2025},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = (
            _load_code_search_code_retrieval(
                path=self.metadata.dataset["path"],
                langs=self.hf_subsets,
                splits=self.metadata.eval_splits,
                revision=self.metadata.dataset["revision"],
            )
        )

        self.data_loaded = True
