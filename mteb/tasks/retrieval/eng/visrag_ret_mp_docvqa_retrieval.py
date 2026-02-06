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


class VisRAGRetMPDocVQA(AbsTaskRetrieval):
    """VisRAG Retrieval task for MP-DocVQA industrial documents.

    The corpus contains scanned pages from multi‑page industrial documents and
    the queries are questions targeting specific pages.  Each query has one
    relevant page image.
    """

    metadata = TaskMetadata(
        name="VisRAGRetMPDocVQA",
        description=(
            "Retrieve scanned document pages based on question prompts.  "
            "The corpus consists of 741 page images from multi-page industrial "
            "documents and the 591 queries originate from the MP-DocVQA dataset.  "
            "Each query maps to exactly one relevant page image."
        ),
        reference="https://arxiv.org/abs/2212.05935",
        type="Retrieval",
        task_subtypes=["Image Text Retrieval"],
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "openbmb/VisRAG-Ret-Test-MP-DocVQA",
            "revision": "3ebd091c458cf04161f78cd7b12ea101f83e2529",
        },
        date=("1900-01-01", "2020-12-31"),
        domains=["Engineering", "Non-fiction"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{tito2023hierarchicalmultimodaltransformersmultipage,
      title={Hierarchical multimodal transformers for Multi-Page DocVQA}, 
      author={Rubèn Tito and Dimosthenis Karatzas and Ernest Valveny},
      year={2023},
      eprint={2212.05935},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2212.05935}, 
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
