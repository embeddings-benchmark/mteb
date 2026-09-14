"""Local, deliberately unregistered NQ-Tables engineering prototype.

Import this class explicitly; it is not exported by ``mteb.tasks``. The source
license discrepancy remains unresolved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset

    from mteb.timing import TimingStack


class NQTablesRetrieval(AbsTaskRetrieval):
    """Retrieve the released Markdown tables using the original release qrels."""

    supported_splits = ("train", "dev", "test")

    metadata = TaskMetadata(
        name="NQTablesRetrieval",
        dataset={
            "path": "ibm-research/NQTablesRetrieval",
            "revision": "4962c33f5e651c82bc82c013893061202a47685e",
        },
        description=(
            "This task preserves original NQ-Tables qrels and is an engineering "
            "baseline, not the final table-necessary OmniWikipedia benchmark. "
            "Local prototype; dataset license discrepancy remains unresolved."
        ),
        reference="https://aclanthology.org/2021.naacl-main.43/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        domains=["Written", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        license=None,  # Intentionally unresolved; not a publication-ready task.
        annotations_creators="derived",
        sample_creation="found",
        is_beta=True,
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Retain source rows as well as MTEB's mapping representation.
        self.source_qrels: dict[str, Dataset] = {}
        self._shared_corpus: Dataset | None = None

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        """Load requested splits without filtering, deduplicating or rewriting text.

        Defaults to test. Use ``filter_eval_splits(["train", "dev", "test"])``
        before loading to inspect all splits. Additional calls can load further
        splits while reusing the same corpus. ``kwargs`` go to ``load_dataset``
        (for example, ``cache_dir``); the source and revision stay pinned.
        """
        unknown = set(self.eval_splits) - set(self.supported_splits)
        if unknown:
            raise ValueError(f"Unsupported NQ-Tables splits: {sorted(unknown)}")

        if self._shared_corpus is None:
            corpus = load_dataset(
                **self.metadata.dataset,
                name="corpus_md",
                split="corpus_md",
                num_proc=num_proc,
                **kwargs,
            ).rename_column("_id", "id")
            if len(set(corpus["id"])) != len(corpus):
                raise ValueError("Duplicate corpus IDs cannot be represented safely")
            self._shared_corpus = corpus

        corpus = self._shared_corpus
        if self.dataset is None:
            self.dataset = {"default": {}}
        for split in self.eval_splits:
            if split in self.dataset["default"]:
                continue
            queries = load_dataset(
                **self.metadata.dataset,
                name="queries",
                split=f"{split}_queries",
                num_proc=num_proc,
                **kwargs,
            ).rename_column("_id", "id")
            query_texts = {}
            for row in queries:
                qid, text = row["id"], row["text"]
                if qid in query_texts and query_texts[qid] != text:
                    raise ValueError(f"Conflicting query text for ID {qid!r}")
                query_texts[qid] = text

            qrels = load_dataset(
                **self.metadata.dataset,
                name="default",
                split=split,
                num_proc=num_proc,
                **kwargs,
            )
            relevant_docs: dict[str, dict[str, int]] = {}
            for row in qrels:
                qid, did, score = row["qid"], row["did"], row["score"]
                docs = relevant_docs.setdefault(qid, {})
                if did in docs:
                    raise ValueError(f"Repeated qrel pair in {split}: {qid!r}, {did!r}")
                docs[did] = score

            self.source_qrels[split] = qrels
            self.dataset["default"][split] = {
                "corpus": corpus,
                "queries": queries,
                "relevant_docs": relevant_docs,
                "top_ranked": None,
            }
        self.data_loaded = True
