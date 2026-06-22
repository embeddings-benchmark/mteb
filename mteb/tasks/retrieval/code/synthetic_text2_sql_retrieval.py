from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"

_METADATA = dict(
    description="The dataset is a collection of natural language queries and their corresponding sql snippets. The task is to retrieve the most relevant code snippet for a given query.",
    reference="https://huggingface.co/datasets/gretelai/synthetic_text_to_sql",
    dataset={
        "path": "CoIR-Retrieval/synthetic-text2sql",
        "revision": "686b87296c3a0191b5d9415a00526c62db9fce09",
    },
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=[_EVAL_SPLIT],
    eval_langs=["eng-Latn", "sql-Code"],
    main_score="ndcg_at_10",
    date=("2019-01-01", "2019-12-31"),
    domains=["Programming", "Written"],
    task_subtypes=["Code retrieval"],
    license="mit",
    annotations_creators="derived",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@software{gretel-synthetic-text-to-sql-2024,
  author = {Meyer, Yev and Emadi, Marjan and Nathawani, Dhruv and Ramaswamy, Lipika and Boyd, Kendrick and Van Segbroeck, Maarten and Grossman, Matthew and Mlocek, Piotr and Newberry, Drew},
  month = {April},
  title = {{Synthetic-Text-To-SQL}: A synthetic dataset for training language models to generate SQL queries from natural language prompts},
  url = {https://huggingface.co/datasets/gretelai/synthetic-text-to-sql},
  year = {2024},
}
""",
)


class SyntheticText2SQLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SyntheticText2SQL",
        superseded_by="SyntheticText2SQL.v2",
        **_METADATA,
    )


class SyntheticText2SQLRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SyntheticText2SQL.v2",
        **_METADATA,
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        """Drop corpus documents that are exact duplicates of a relevant document.

        The published corpus pools the train and test partitions, and >94% of the
        evaluated documents have an identical-text twin in the train partition under
        a different ID. Because the qrels only reference the test ID, retrieving the
        identical train document is wrongly scored as a false positive, which tanks
        the metrics (see issue #4626). Removing these duplicate twins restores correct
        scoring without altering any document that is actually referenced by the qrels.
        """
        for subset in self.dataset:
            for split in self.dataset[subset]:
                data = self.dataset[subset][split]
                corpus = data["corpus"]

                relevant_ids = {
                    doc_id for docs in data["relevant_docs"].values() for doc_id in docs
                }
                content_cols = [c for c in corpus.column_names if c != "id"]
                signatures = list(
                    zip(*(corpus[c] for c in content_cols))
                    if content_cols
                    else ([()] * len(corpus))
                )
                relevant_signatures = {
                    sig
                    for doc_id, sig in zip(corpus["id"], signatures)
                    if doc_id in relevant_ids
                }
                drop_ids = {
                    doc_id
                    for doc_id, sig in zip(corpus["id"], signatures)
                    if doc_id not in relevant_ids and sig in relevant_signatures
                }
                if drop_ids:
                    data["corpus"] = corpus.filter(
                        lambda row: row["id"] not in drop_ids,
                        num_proc=num_proc,
                    )
