"""NQ-Tables retrieval task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from mteb.timing import TimingStack


_BIBTEX = r"""
@misc{doshi2026tableir,
  author = {Doshi, Meet and Boni, Odellia and Kumar, Vishwajeet and Sen, Jaydeep and Joshi, Sachindra},
  howpublished = {https://huggingface.co/collections/ibm-research/table-text-ir-evaluation},
  institution = {IBM Research},
  note = {Hugging Face dataset collection},
  title = {TableIR Eval: Table-Text IR Evaluation Collection},
  year = {2026},
}

@inproceedings{herzig-etal-2021-open,
  address = {Online},
  author = {Herzig, Jonathan and M{\"u}ller, Thomas and Krichene, Syrine and Eisenschlos, Julian Martin},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  doi = {10.18653/v1/2021.naacl-main.43},
  month = jun,
  pages = {512--519},
  publisher = {Association for Computational Linguistics},
  title = {Open Domain Question Answering over Tables via Dense Retrieval},
  url = {https://aclanthology.org/2021.naacl-main.43/},
  year = {2021},
}
"""


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
            "Retrieve the reference Wikipedia table associated with a Natural "
            "Questions query under the original NQ-Tables relevance judgments. "
            "The task uses the Markdown table serialization, questions, stable "
            "source IDs, and qrels from the NQ-Tables TableIR release."
        ),
        reference="https://aclanthology.org/2021.naacl-main.43/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),  # publication year
        domains=["Written", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        adapted_from=["NQ"],
        prompt={"query": "Given a question, retrieve its reference Wikipedia table."},
    )

    def load_data(
        self,
        num_proc: int | None = None,
        *,
        timer: TimingStack | None = None,
        **kwargs: Any,
    ) -> None:
        """Load requested splits without filtering, deduplicating or rewriting text.

        Defaults to test. Use ``filter_eval_splits(["train", "dev", "test"])``
        before loading to inspect all splits with the shared corpus. ``kwargs``
        go to ``load_dataset`` (for example, ``cache_dir``); the source and
        revision stay pinned.
        """
        unknown = set(self.eval_splits) - set(self.supported_splits)
        if unknown:
            raise ValueError(f"Unsupported NQ-Tables splits: {sorted(unknown)}")

        if self.data_loaded:
            return

        corpus = load_dataset(
            **self.metadata.dataset,
            name="corpus_md",
            split="corpus_md",
            num_proc=num_proc,
            **kwargs,
        ).rename_column("_id", "id")
        if len(set(corpus["id"])) != len(corpus):
            raise ValueError("Duplicate corpus IDs cannot be represented safely")

        self.dataset = {"default": {}}
        for split in self.eval_splits:
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

            self.dataset["default"][split] = {
                "corpus": corpus,
                "queries": queries,
                "relevant_docs": relevant_docs,
                "top_ranked": None,
            }
        self.data_loaded = True
