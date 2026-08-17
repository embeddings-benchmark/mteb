from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "myang333/CaReBench"
_DATASET_REVISION = "2871160f7089bc22d1f9c16f1af43e11a80efb91"
_BIBTEX = r"""
@article{xu2025carebench,
  author = {Xu, Yifan and Li, Xinhao and Yang, Yichun and Meng, Desen and Huang, Rui and Wang, Limin},
  journal = {arXiv preprint arXiv:2501.00513},
  title = {CaReBench: A Fine-grained Benchmark for Video Captioning and Retrieval},
  year = {2025},
}
"""


def _load_carebench(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all CaReBench retrieval directions.

    TODO: Reupload dataset in standard format and remove this custom load_data.
    """
    if task.data_loaded:
        return
    task.dataset = {"default": {}}
    dataset = load_dataset(
        task.metadata.dataset["path"],
        revision=task.metadata.dataset["revision"],
        split=task.metadata.eval_splits[0],
    )
    dataset = dataset.add_column("id", [str(i) for i in range(len(dataset))])

    query = dataset.select_columns(["id"] + query_columns)
    corpus = dataset.select_columns(["id"] + corpus_columns)
    if "caption" in query_columns:
        query = query.rename_column("caption", "text")
    if "caption" in corpus_columns:
        corpus = corpus.rename_column("caption", "text")

    qrels = {str(i): {str(i): 1} for i in range(len(dataset))}
    task.dataset["default"]["test"] = RetrievalSplitData(
        queries=query, corpus=corpus, relevant_docs=qrels, top_ranked=None
    )
    task.data_loaded = True


class CaReBenchT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CaReBenchT2VRetrieval",
        description=(
            "Retrieve the video clip that matches a given fine-grained English "
            "caption from CaReBench, 1,000 videos with detailed human-annotated "
            "captions (~228 words on average) covering both static objects and "
            "dynamic actions."
            " Used the `CaReBench` config test split (1,000 examples)."
        ),
        reference="https://arxiv.org/abs/2501.00513",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2021-01-01", "2024-12-31"),
        domains=["Activity", "Web"],
        task_subtypes=["Caption Pairing"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_carebench(self, query_columns=["caption"], corpus_columns=["video"])


class CaReBenchV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CaReBenchV2TRetrieval",
        description=(
            "Retrieve the fine-grained English caption that describes a given video "
            "from CaReBench, 1,000 videos with detailed human-annotated captions "
            "(~228 words on average) covering both static objects and dynamic "
            "actions."
            " Used the `CaReBench` config test split (1,000 examples)."
        ),
        reference="https://arxiv.org/abs/2501.00513",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2021-01-01", "2024-12-31"),
        domains=["Activity", "Web"],
        task_subtypes=["Caption Pairing"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that best describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_carebench(self, query_columns=["video"], corpus_columns=["caption"])
