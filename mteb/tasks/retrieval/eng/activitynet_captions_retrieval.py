from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/ActivityNet_Captions_val2"
_DATASET_REVISION = "e87473c4832ea982bbeca1dde94bbebfa6ea6ada"
_BIBTEX = r"""
@inproceedings{krishna2017dense,
  author = {Krishna, Ranjay and Hata, Kenji and Ren, Frederic and Fei-Fei, Li and Niebles, Juan Carlos},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  title = {Dense-Captioning Events in Videos},
  year = {2017},
}
"""


def _load_activitynet(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all ActivityNet Captions retrieval directions.

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


class ActivityNetCaptionsV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ActivityNetCaptionsV2TRetrieval",
        description=(
            "Retrieve the English caption that describes a given video clip from "
            "ActivityNet Captions. Each example pairs one video with one reference "
            "description (1:1 retrieval)."
        ),
        reference="https://huggingface.co/datasets/mteb/ActivityNet_Captions_val2",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that best describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_activitynet(self, query_columns=["video"], corpus_columns=["caption"])


class ActivityNetCaptionsT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ActivityNetCaptionsT2VRetrieval",
        description=(
            "Retrieve the video clip that matches a given English caption from "
            "ActivityNet Captions. Each example pairs one caption with one reference "
            "video (1:1 retrieval)."
        ),
        reference="https://huggingface.co/datasets/mteb/ActivityNet_Captions_val2",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_activitynet(self, query_columns=["caption"], corpus_columns=["video"])
