from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _load_visrag_data(
    path: str,
    splits: list[str],
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
            remove_columns=["query-id", "query", "answer", "options", "is_numerical"],
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
                "image": x["image"],
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


class VisRAGRetChartQA(AbsTaskRetrieval):
    """VisRAG Retrieval task for ChartQA charts.

    The corpus contains chart images and the queries are multiple‑choice questions
    about those charts.  Each query is linked to one relevant chart image.
    """

    metadata = TaskMetadata(
        name="VisRAGRetChartQA",
        description="Assess end-to-end vision-based RAG performance on real-world charts requiring complex logical and visual reasoning from retrieved images.",
        reference="https://arxiv.org/abs/2203.10244",
        type="Retrieval",
        task_subtypes=["Image Text Retrieval"],
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "mteb/VisRAGRetChartQA",
            "revision": "4665d4a3ebc9ea5a4e635985a4fd2a595b78b3c0",
        },
        date=("2010-01-01", "2021-12-31"),
        domains=["Web", "Non-fiction"],
        license="gpl-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{masry2022chartqabenchmarkquestionanswering,
  archiveprefix = {arXiv},
  author = {Ahmed Masry and Do Xuan Long and Jia Qing Tan and Shafiq Joty and Enamul Hoque},
  eprint = {2203.10244},
  primaryclass = {cs.CL},
  title = {ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning},
  url = {https://arxiv.org/abs/2203.10244},
  year = {2022},
}""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_visrag_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
