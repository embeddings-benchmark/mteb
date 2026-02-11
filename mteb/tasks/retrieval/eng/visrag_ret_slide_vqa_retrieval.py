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


class VisRAGRetSlideVQA(AbsTaskRetrieval):
    """VisRAG Retrieval task for SlideVQA slide decks.

    The corpus contains slide images from educational slide decks and the queries
    are questions about the slides.  Each query has one relevant slide image.
    """

    metadata = TaskMetadata(
        name="VisRAGRetSlideVQA",
        description="Retrieve and reason across multiple slide images within a deck to answer multi-hop questions in a vision-centric retrieval-augmented generation pipeline.",
        reference="https://arxiv.org/abs/2301.04883",
        type="Retrieval",
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "mteb/VisRAGRetSlideVQA",
            "revision": "c62fb65928b0bf7b709cd3084c87edf75a1ba29b",
        },
        date=("2010-01-01", "2022-12-31"),
        domains=["Web"],
        task_subtypes=["Image Text Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{tanaka2023slidevqadatasetdocumentvisual,
  archiveprefix = {arXiv},
  author = {Ryota Tanaka and Kyosuke Nishida and Kosuke Nishida and Taku Hasegawa and Itsumi Saito and Kuniko Saito},
  eprint = {2301.04883},
  primaryclass = {cs.CL},
  title = {SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images},
  url = {https://arxiv.org/abs/2301.04883},
  year = {2023},
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
