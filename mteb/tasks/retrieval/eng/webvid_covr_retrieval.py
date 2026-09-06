from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class WebVidCoVRIT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WebVidCoVRIT2VRetrieval",
        description=(
            "WebVid-CoVR is a dataset for Compositional Video Retrieval (CoVR) "
            "where the query consists of a reference image (middle frame of reference video) "
            "and an editing instruction. The corpus consists of candidate videos."
        ),
        reference="https://arxiv.org/abs/2308.14746",
        dataset={
            "path": "deep9539/WebVid-CoVR",
            "revision": "22f6c8c41e0e3a2b969be0e58df5217d7059ec1c",
        },
        type="Any2AnyRetrieval",
        category="it2v",
        modalities=["image", "text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{ventura23covr,
  author = {Lucas Ventura and Cordelia Schmid and Gregory Rogez},
  booktitle = {CVPR},
  title = {COVR: Compositional Video Retrieval},
  year = {2023},
}
""",
        prompt={
            "query": "Given the reference image and editing text, retrieve the video that matches the composed query."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return
        path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]
        corpus = load_dataset(path, "corpus", split="test", revision=revision)
        queries = load_dataset(path, "queries", split="test", revision=revision)
        qrels_ds = load_dataset(path, "qrels", split="test", revision=revision)
        qrels: dict[str, dict[str, int]] = {}
        for row in qrels_ds:
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    corpus=corpus, queries=queries, relevant_docs=qrels, top_ranked=None
                )
            }
        }
        self.data_loaded = True
