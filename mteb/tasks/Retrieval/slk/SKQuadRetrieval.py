from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


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
        category="s2s",
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
            self.corpus = {
                "test": {
                    row["_id"]: {"text": row["text"], "title": row["title"]}
                    for row in ds_corpus["corpus"]
                }
            }
            self.queries = {
                "test": {row["_id"]: row["text"] for row in ds_query["queries"]}
            }
            self.relevant_docs = {"test": {}}

            for row in ds_default["test"]:
                self.relevant_docs["test"].setdefault(row["query-id"], {})[
                    row["corpus-id"]
                ] = int(row["score"])

            print(
                f"Data Loaded:\n- Corpus size: {len(self.corpus['test'])}\n- Query size: {len(self.queries['test'])}\n- Relevance entries: {len(self.relevant_docs['test'])}"
            )
