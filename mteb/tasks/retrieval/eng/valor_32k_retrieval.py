from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/VALOR-32K"
_DATASET_REVISION = "003f74e0b5031c81bc0f3416f5d3de1153e21e16"
_BIBTEX = r"""
@article{chen2023valor,
  author = {Chen, Sihan and He, Xingjian and Guo, Longteng and Zhu, Xinxin and Wang, Weining and Tang, Jinhui and Liu, Jing},
  title = {VALOR: Vision-Audio-Language Omni-perception Pretraining Model and Dataset},
  journal = {arXiv preprint arXiv:2304.08345},
  year = {2023},
}
"""


def _load_valor(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all VALOR-32K retrieval directions.

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
    if "description" in query_columns:
        query = query.rename_column("description", "text")
    if "description" in corpus_columns:
        corpus = corpus.rename_column("description", "text")

    qrels = {str(i): {str(i): 1} for i in range(len(dataset))}
    task.dataset["default"]["test"] = RetrievalSplitData(
        queries=query, corpus=corpus, relevant_docs=qrels, top_ranked=None
    )
    task.data_loaded = True


class VALOR32KV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VALOR32KV2TRetrieval",
        description=(
            "Retrieve the description that matches a given video clip (video only) "
            "from the VALOR-32K benchmark of vision-audio-language understanding."
        ),
        reference="https://arxiv.org/abs/2304.08345",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2023-04-01", "2023-04-30"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the description that matches the following video."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_valor(self, query_columns=["video"], corpus_columns=["description"])


class VALOR32KT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VALOR32KT2VRetrieval",
        description=(
            "Retrieve the video clip (video only) that matches a given description "
            "from the VALOR-32K benchmark of vision-audio-language understanding."
        ),
        reference="https://arxiv.org/abs/2304.08345",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2023-04-01", "2023-04-30"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given description."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_valor(self, query_columns=["description"], corpus_columns=["video"])


class VALOR32KVA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VALOR32KVA2TRetrieval",
        description=(
            "Retrieve the description that matches a given video+audio clip "
            "from the VALOR-32K benchmark of vision-audio-language understanding."
        ),
        reference="https://arxiv.org/abs/2304.08345",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video", "text"],
        date=("2023-04-01", "2023-04-30"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the description that matches the following video."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_valor(
            self, query_columns=["video", "audio"], corpus_columns=["description"]
        )


class VALOR32KT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VALOR32KT2VARetrieval",
        description=(
            "Retrieve the video+audio clip that matches a given description "
            "from the VALOR-32K benchmark of vision-audio-language understanding."
        ),
        reference="https://arxiv.org/abs/2304.08345",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "audio", "video"],
        date=("2023-04-01", "2023-04-30"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given description."},
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_valor(
            self, query_columns=["description"], corpus_columns=["video", "audio"]
        )
