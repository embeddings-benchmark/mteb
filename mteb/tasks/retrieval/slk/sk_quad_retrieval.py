import logging

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)


class SKQuadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SKQuadRetrieval",
        description=(
            "Retrieval SK Quad evaluates Slovak search performance using questions and answers "
            "derived from the SK-QuAD dataset. It measures relevance with scores assigned to answers "
            "based on their relevancy to corresponding questions, which is vital for improving "
            "Slovak language search systems."
        ),
        reference="https://huggingface.co/datasets/TUKE-KEMT/retrieval-skquad",
        dataset={
            "path": "TUKE-KEMT/retrieval-skquad",
            "revision": "09f81f51dd5b8497da16d02c69c98d5cb5993ef2",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="ndcg_at_10",
        date=("2024-05-30", "2024-06-13"),
        domains=["Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def load_data(self, eval_splits=None, **kwargs):
        """Load and preprocess datasets for retrieval task."""
        eval_splits = eval_splits or ["test"]

        # Load datasets
        ds_default = load_dataset("TUKE-KEMT/retrieval-skquad", "default")
        ds_corpus = load_dataset("TUKE-KEMT/retrieval-skquad", "corpus")
        ds_query = load_dataset("TUKE-KEMT/retrieval-skquad", "queries")

        if "test" in eval_splits:
            # Corpus, Queries, and Relevance dictionary for 'test' split
            corpus_dict = {
                row["_id"]: {"text": row["text"], "title": row["title"]}
                for row in ds_corpus["corpus"]
            }
            queries_dict = {row["_id"]: row["text"] for row in ds_query["queries"]}
            relevant_docs = {}

            for row in ds_default["test"]:
                relevant_docs.setdefault(row["query-id"], {})[row["corpus-id"]] = int(
                    row["score"]
                )

            corpus_dataset = Dataset.from_list(
                [
                    {"id": k, "text": v["text"], "title": v["title"]}
                    for k, v in corpus_dict.items()
                ]
            )
            queries_dataset = Dataset.from_list(
                [{"id": k, "text": v} for k, v in queries_dict.items()]
            )

            self.dataset = {
                "default": {
                    "test": {
                        "corpus": corpus_dataset,
                        "queries": queries_dataset,
                        "relevant_docs": relevant_docs,
                        "top_ranked": None,
                    }
                }
            }

            logger.info(
                f"Data Loaded:\n- Corpus size: {len(corpus_dict)}\n- Query size: {len(queries_dict)}\n- Relevance entries: {len(relevant_docs)}"
            )
