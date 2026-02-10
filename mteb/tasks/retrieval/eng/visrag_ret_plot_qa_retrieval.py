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


class VisRAGRetPlotQA(AbsTaskRetrieval):
    """VisRAG Retrieval task for PlotQA scientific plots.

    The corpus contains scientific plot images and the queries are questions
    requiring reasoning over the plots.  Each query has exactly one relevant
    plot image.
    """

    metadata = TaskMetadata(
        name="VisRAGRetPlotQA",
        description="Execute vision-based retrieval and numerical reasoning over scientific plots to answer questions without relying on structured data parsing.",
        reference="https://arxiv.org/abs/1909.00997",
        type="Retrieval",
        task_subtypes=["Image Text Retrieval"],
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "openbmb/VisRAG-Ret-Test-PlotQA",
            "revision": "ef953ef8ab6d78ac112dd4cde6acdb2c2692039a",
        },
        date=("2000-01-01", "2019-12-31"),
        domains=["Web", "Non-fiction"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{methani2020plotqareasoningscientificplots,
      title={PlotQA: Reasoning over Scientific Plots}, 
      author={Nitesh Methani and Pritha Ganguly and Mitesh M. Khapra and Pratyush Kumar},
      year={2020},
      eprint={1909.00997},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1909.00997},
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
