from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/panda-70m"
_DATASET_REVISION = "62c3a1ae9adbaae91ca1ca6b0b7adcbf8f36409f"
_BIBTEX = r"""
@inproceedings{chen2024panda,
  author = {Chen, Tsai-Shien and Siarohin, Aliaksandr and Menapace, Willi and Deyneka, Ekaterina and Chao, Hsiang-wei and Jeon, Byung Eun and Fang, Yuwei and Lee, Hsin-Ying and Ren, Jian and Yang, Ming-Hsuan and Tulyakov, Sergey},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers},
  year = {2024},
}
"""


def _load_panda70m(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all Panda-70M retrieval directions.

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


class Panda70MV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Panda70MV2TRetrieval",
        description=(
            "Retrieve the English caption that describes a given video clip (video "
            "only) from Panda-70M, a large-scale video captioning dataset sourced "
            "from YouTube."
        ),
        reference="https://arxiv.org/abs/2402.19479",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2024-01-01", "2024-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_panda70m(self, query_columns=["video"], corpus_columns=["caption"])


class Panda70MT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Panda70MT2VRetrieval",
        description=(
            "Retrieve the video clip (video only) that matches a given English "
            "caption from Panda-70M, a large-scale video captioning dataset sourced "
            "from YouTube."
        ),
        reference="https://arxiv.org/abs/2402.19479",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2024-01-01", "2024-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_panda70m(self, query_columns=["caption"], corpus_columns=["video"])


class Panda70MVA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Panda70MVA2TRetrieval",
        description=(
            "Retrieve the English caption that describes a given video clip from "
            "Panda-70M, a large-scale video captioning dataset sourced from YouTube."
        ),
        reference="https://arxiv.org/abs/2402.19479",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video", "text"],
        date=("2024-01-01", "2024-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_panda70m(
            self, query_columns=["video", "audio"], corpus_columns=["caption"]
        )


class Panda70MT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Panda70MT2VARetrieval",
        description=(
            "Retrieve the video clip that matches a given English caption from "
            "Panda-70M, a large-scale video captioning dataset sourced from YouTube."
        ),
        reference="https://arxiv.org/abs/2402.19479",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "audio", "video"],
        date=("2024-01-01", "2024-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_panda70m(
            self, query_columns=["caption"], corpus_columns=["video", "audio"]
        )
