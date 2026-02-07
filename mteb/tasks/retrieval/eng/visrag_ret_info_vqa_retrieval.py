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


class VisRAGRetInfoVQA(AbsTaskRetrieval):
    """VisRAG Retrieval task for InfoVQA infographics.

    The corpus contains infographic images and the queries are questions about
    those infographics.  Each query points to one relevant infographic image.
    """

    metadata = TaskMetadata(
        name="VisRAGRetInfoVQA",
        description=(
            "Retrieve infographics based on natural‑language questions.  "
            "The corpus contains 459 infographic images and the 718 queries "
            "come from the InfographicVQA dataset.  Each query has a single "
            "relevant image."
        ),
        reference="https://arxiv.org/abs/2104.12756",
        type="Retrieval",
        task_subtypes=["Image Text Retrieval"],
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "openbmb/VisRAG-Ret-Test-InfoVQA",
            "revision": "04ee2db1d0a05304899453a3b03baef5e18991a3",
        },
        date=("2010-01-01", "2020-12-31"),
        domains=["Web"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{mathew2021infographicvqa,
      title={InfographicVQA}, 
      author={Minesh Mathew and Viraj Bagal and Rubèn Pérez Tito and Dimosthenis Karatzas and Ernest Valveny and C. V Jawahar},
      year={2021},
      eprint={2104.12756},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2104.12756}, 
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
