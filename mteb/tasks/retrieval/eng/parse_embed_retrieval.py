from __future__ import annotations

from collections import defaultdict
from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class ParseEmbedRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ParseEmbedRetrieval",
        description=(
            "ParseEmbed is a synthetic retrieval benchmark with 720 queries and "
            "2,880 near-duplicate documents. Its meaning, text-formatting, and "
            "table splits test whether models preserve parse-sensitive details "
            "rather than broad topical similarity."
        ),
        reference="https://huggingface.co/datasets/Convence/ParseEmbed",
        dataset={
            "path": "Convence/ParseEmbed",
            "revision": "2616064a8a91e6bbe7e7cdef7c861469b10688a5",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["mean", "text_formatting", "table"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2026-05-21", "2026-05-21"),
        domains=["Constructed", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="apache-2.0",
        annotations_creators="algorithmic",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        prompt={
            "query": (
                "Given a precise retrieval request, retrieve the document that "
                "exactly satisfies every stated condition."
            )
        },
        contributed_by="Convence",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        dataset_path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]
        corpus = load_dataset(
            dataset_path,
            "corpus",
            split="test",
            revision=revision,
            num_proc=num_proc,
        )
        corpus = corpus.rename_column("_id", "id").select_columns(["id", "text"])

        qrels = load_dataset(
            dataset_path,
            split="test",
            revision=revision,
            num_proc=num_proc,
        )
        all_relevant_docs: defaultdict[str, dict[str, int]] = defaultdict(dict)
        for row in qrels:
            all_relevant_docs[row["query-id"]][row["corpus-id"]] = int(row["score"])

        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            task_rows = load_dataset(
                dataset_path,
                "parse-embed",
                split=split,
                revision=revision,
                num_proc=num_proc,
            )
            queries = task_rows.select_columns(["id", "query"]).rename_column(
                "query", "text"
            )
            relevant_docs = {
                query_id: all_relevant_docs[query_id] for query_id in queries["id"]
            }
            self.dataset["default"][split] = RetrievalSplitData(
                corpus=corpus,
                queries=queries,
                relevant_docs=relevant_docs,
                top_ranked=None,
            )

        self.data_loaded = True
