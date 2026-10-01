from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset

    from mteb.types import RelevantDocumentsType


def _load_data(
    path: str,
    splits: list[str],
    revision: str | None = None,
    num_proc: int | None = None,
) -> tuple[dict[str, Dataset], dict[str, Dataset], dict[str, RelevantDocumentsType]]:
    corpus = {}
    queries = {}
    relevant_docs = {}

    for split in splits:
        query_ds = load_dataset(
            path,
            "queries",
            split=split,
            revision=revision,
            num_proc=num_proc,
        )
        queries[split] = query_ds.map(
            lambda row, split=split: {
                "id": f"query-{split}-{row['query-id']}",
                "text": row["query"],
            },
            remove_columns=query_ds.column_names,
            num_proc=num_proc,
        )

        corpus_ds = load_dataset(
            path,
            "corpus",
            split=split,
            revision=revision,
            num_proc=num_proc,
        )
        corpus[split] = corpus_ds.map(
            lambda row, split=split: {
                "id": f"corpus-{split}-{row['corpus-id']}",
            },
            remove_columns=["corpus-id"],
            num_proc=num_proc,
        ).select_columns(["id", "image"])

        qrels_ds = load_dataset(
            path,
            "qrels",
            split=split,
            revision=revision,
            num_proc=num_proc,
        )
        relevant_docs[split] = {}
        for row in qrels_ds:
            query_id = f"query-{split}-{row['query-id']}"
            corpus_id = f"corpus-{split}-{row['corpus-id']}"
            relevant_docs[split].setdefault(query_id, {})[corpus_id] = int(row["score"])

    return corpus, queries, relevant_docs


class MMLongBenchDocRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMLongBenchDocRetrieval",
        description="MMLongBench-Doc is a long-context, multimodal document understanding benchmark built from lengthy PDFs with rich layouts and content including text, tables, charts, and images. This retrieval adaptation asks models to retrieve pages from the source document for a question, assigning higher graded relevance to annotated evidence pages.",
        reference="https://arxiv.org/abs/2407.01523",
        dataset={
            "path": "VLM2Vec/MMLongBench-doc",
            "revision": "1c2ee8f611a6c552d4bd90daca3a92d7a55ff806",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2024-01-01", "2024-07-01"),
        domains=["Academic", "Government", "Legal", "Financial", "Engineering"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="multiple",
        bibtex_citation=r"""
@misc{ma2024mmlongbenchdocbenchmarkinglongcontextdocument,
  archiveprefix = {arXiv},
  author = {Yubo Ma and Yuhang Zang and Liangyu Chen and Meiqi Chen and Yizhu Jiao and Xinze Li and Xinyuan Lu and Ziyu Liu and Yan Ma and Xiaoyi Dong and Pan Zhang and Liangming Pan and Yu-Gang Jiang and Jiaqi Wang and Yixin Cao and Aixin Sun},
  eprint = {2407.01523},
  primaryclass = {cs.CV},
  title = {MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations},
  url = {https://arxiv.org/abs/2407.01523},
  year = {2024},
}
""",
        prompt={"query": "Find a document image that matches the given query."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
            num_proc=num_proc,
        )
        self.data_loaded = True
