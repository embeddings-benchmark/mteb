from __future__ import annotations

from typing import Any

import polars as pl
from datasets import concatenate_datasets, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


def _load_data(
    path: str,
    splits: str,
    revision: str | None = None,
    point_to_kept_copy: bool = False,
) -> dict[str, RetrievalSplitData]:
    corpus = {}
    queries = {}
    relevant_docs = {}

    dataset = load_dataset(
        path,
        revision=revision,
    )
    dataset_splits = ["test", "validation", "train"]
    shared_corpus = concatenate_datasets([dataset[split] for split in dataset_splits])

    text_df = pl.DataFrame({"text": shared_corpus["text"]})
    unique_indices = text_df["text"].arg_unique()
    shared_corpus = shared_corpus.select(unique_indices)
    # the corpus keeps one meme per text; v2 points each query at that meme
    kept_id = {
        text: "corpus-" + str(row_id)
        for text, row_id in zip(shared_corpus["text"], shared_corpus["id"], strict=True)
    }

    shared_corpus = shared_corpus.map(
        lambda x: {
            "id": "corpus-" + str(x["id"]),
            "modality": "image",
        },
        remove_columns=[
            "split",
            "label",
        ],
    )

    for split in splits:
        corpus[split] = shared_corpus
        split_dataset = dataset[split]
        queries[split] = split_dataset.map(
            lambda x: {
                "id": "query-" + str(x["id"]),
                "modality": "text",
            },
            remove_columns=[
                "split",
                "label",
            ],
        )
        relevant_docs[split] = {}
        for example in split_dataset:
            query_id = "query-" + str(example["id"])
            doc_id = (
                kept_id[example["text"]]
                if point_to_kept_copy
                else "corpus-" + str(example["id"])
            )
            if query_id not in relevant_docs[split]:
                relevant_docs[split][query_id] = {}
            relevant_docs[split][query_id][doc_id] = 1

    return {
        split: RetrievalSplitData(
            corpus=corpus[split],
            queries=queries[split],
            relevant_docs=relevant_docs[split],
            top_ranked=None,
        )
        for split in splits
    }


class HatefulMemesT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HatefulMemesT2IRetrieval",
        description="Retrieve captions based on memes to assess OCR abilities.",
        reference="https://arxiv.org/pdf/2005.04790",
        dataset={
            "path": "mteb/HatefulMemesT2IRetrieval",
            "revision": "25492b29a8e7455c6193481b896bced17e0be796",
        },
        type="Any2AnyRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        superseded_by="HatefulMemesT2IRetrieval.v2",
        bibtex_citation=r"""
@article{kiela2020hateful,
  author = {Kiela, Douwe and Firooz, Hamed and Mohan, Aravind and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Pratik and Testuggine, Davide},
  journal = {Advances in neural information processing systems},
  pages = {2611--2624},
  title = {The hateful memes challenge: Detecting hate speech in multimodal memes},
  volume = {33},
  year = {2020},
}
""",
    )


class HatefulMemesT2IRetrievalV2(AbsTaskRetrieval):
    metadata = HatefulMemesT2IRetrieval.metadata.model_copy(
        update={
            "dataset": {
                "path": "mteb/MMSoc_HatefulMemes",
                "revision": "852290806bbf9d660f5c04c246f81ce38bfc1d48",
            },
            "name": "HatefulMemesT2IRetrieval.v2",
            "description": HatefulMemesT2IRetrieval.metadata.description
            + " Version 2 points each query at the corpus meme kept for its text: the corpus keeps one meme per text, and v1 pointed 171 of the 1,000 test queries at removed duplicates.",
            "adapted_from": ["HatefulMemesT2IRetrieval"],
            "superseded_by": None,
        }
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        self.dataset = {
            "default": _load_data(
                path=self.metadata.dataset["path"],
                splits=self.metadata.eval_splits,
                revision=self.metadata.dataset["revision"],
                point_to_kept_copy=True,
            )
        }
        self.data_loaded = True
