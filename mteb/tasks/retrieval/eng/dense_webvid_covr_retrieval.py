from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class DenseWebVidCoVRVT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DenseWebVidCoVRVT2VRetrieval",
        description=(
            "Dense-WebVid-CoVR is a dataset for Compositional Video Retrieval (CoVR) "
            "where the query consists of a reference video and an editing instruction. "
            "The corpus consists of candidate videos."
        ),
        reference="https://arxiv.org/abs/2508.14039",
        dataset={
            "path": "nik1995/Dense-WebVid-CoVR",
            "revision": "main",
        },
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{thawakar2025bse,
  title={BSE-CoVR: Broadening Semantic Edit Support for Compositional Video Retrieval},
  author={Thawakar, Omkar and others},
  journal={arXiv preprint arXiv:2508.14039},
  year={2025}
}
""",
        prompt={
            "query": "Given the reference video and editing text, retrieve the video that matches the composed query."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
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
