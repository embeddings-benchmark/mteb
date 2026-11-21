from __future__ import annotations

from collections import defaultdict
from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


def load_retrieval_data(
    dataset_path: str, eval_splits: list[str]
) -> dict[str, RetrievalSplitData]:
    eval_split = eval_splits[0]
    corpus_dataset = load_dataset(dataset_path, "corpus")
    queries_dataset = load_dataset(dataset_path, "queries")
    qrels = load_dataset(dataset_path + "-qrels")[eval_split]

    corpus = (
        corpus_dataset["corpus"]
        .select_columns(["_id", "text"])
        .rename_column("_id", "id")
    )
    queries = (
        queries_dataset["queries"]
        .select_columns(["_id", "text"])
        .rename_column("_id", "id")
    )
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e["query-id"]][e["corpus-id"]] = e["score"]
    return {
        eval_split: RetrievalSplitData(
            corpus=corpus, queries=queries, relevant_docs=relevant_docs, top_ranked=None
        )
    }


class GermanQuADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GermanQuAD-Retrieval",
        description="Context Retrieval for German Question Answering",
        reference="https://huggingface.co/datasets/deepset/germanquad",
        dataset={
            "path": "mteb/germanquad-retrieval",
            "revision": "f5c87ae5a2e7a5106606314eef45255f03151bb3",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="mrr_at_5",
        date=("2020-05-19", "2021-04-26"),
        domains=["Written", "Non-fiction", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{möller2021germanquad,
  address = {Punta Cana, Dominican Republic},
  author = {M{\"o}ller, Timo and Risch, Julian and Pietsch, Malte},
  booktitle = {Proceedings of the 3rd Workshop on Machine Reading for Question Answering},
  doi = {10.18653/v1/2021.mrqa-1.4},
  editor = {Fisch, Adam and Talmor, Alon and Chen, Danqi and Choi, Eunsol and Seo, Minjoon and Lewis, Patrick and Jia, Robin and Min, Sewon},
  month = {nov},
  pages = {42--50},
  publisher = {Association for Computational Linguistics},
  title = {{G}erman{Q}u{AD} and {G}erman{DPR}: Improving Non-{E}nglish Question Answering and Passage Retrieval},
  url = {https://aclanthology.org/2021.mrqa-1.4/},
  year = {2021},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {
            "default": load_retrieval_data(
                self.metadata.dataset["path"], self.metadata.eval_splits
            )
        }
        self.data_loaded = True
