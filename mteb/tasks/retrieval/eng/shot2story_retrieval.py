from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/Shot2Story20K_test"
_DATASET_REVISION = "1957723c0ad97993f3807df271e2660b869c3ea2"
_BIBTEX = r"""
@article{han2023shot2story,
  author = {Han, Mingfei and Zhang, Linjie and Du, Yali and Luo, Junbin and Wang, Xiaodan},
  title = {Shot2Story: A New Benchmark for Comprehensive Understanding of Multi-shot Videos},
  journal = {arXiv preprint arXiv:2312.10300},
  year = {2023},
}
"""


def _load_shot2story(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all Shot2Story20K retrieval directions.

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


class Shot2Story20KV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Shot2Story20KV2TRetrieval",
        description=(
            "Retrieve the detailed summary caption that describes a given video "
            "clip (video only) from the Shot2Story20K benchmark of multi-shot videos."
        ),
        reference="https://arxiv.org/abs/2312.10300",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2023-12-01", "2023-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_shot2story(self, query_columns=["video"], corpus_columns=["caption"])


class Shot2Story20KT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Shot2Story20KT2VRetrieval",
        description=(
            "Retrieve the video clip (video only) that matches a given detailed "
            "summary caption from the Shot2Story20K benchmark of multi-shot videos."
        ),
        reference="https://arxiv.org/abs/2312.10300",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2023-12-01", "2023-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_shot2story(self, query_columns=["caption"], corpus_columns=["video"])


class Shot2Story20KVA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Shot2Story20KVA2TRetrieval",
        description=(
            "Retrieve the detailed summary caption that describes a given video "
            "clip from the Shot2Story20K benchmark of multi-shot videos."
        ),
        reference="https://arxiv.org/abs/2312.10300",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video", "text"],
        date=("2023-12-01", "2023-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_shot2story(
            self, query_columns=["video", "audio"], corpus_columns=["caption"]
        )


class Shot2Story20KT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Shot2Story20KT2VARetrieval",
        description=(
            "Retrieve the video clip that matches a given detailed summary caption "
            "from the Shot2Story20K benchmark of multi-shot videos."
        ),
        reference="https://arxiv.org/abs/2312.10300",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "audio", "video"],
        date=("2023-12-01", "2023-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video clip that matches the given caption."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_shot2story(
            self, query_columns=["caption"], corpus_columns=["video", "audio"]
        )


class Shot2Story20KV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Shot2Story20KV2ARetrieval",
        description=(
            "Retrieve the audio track that matches a given video clip from the "
            "Shot2Story dataset. Tests cross-modal alignment between video frames "
            "and audio."
        ),
        reference="https://arxiv.org/abs/2312.10300",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "audio"],
        date=("2023-12-01", "2023-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the audio that corresponds to the following video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_shot2story(self, query_columns=["video"], corpus_columns=["audio"])


class Shot2Story20KA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Shot2Story20KA2VRetrieval",
        description=(
            "Retrieve the video clip that matches a given audio track from the "
            "Shot2Story dataset. Tests cross-modal alignment between audio and "
            "video frames."
        ),
        reference="https://arxiv.org/abs/2312.10300",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="a2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video"],
        date=("2023-12-01", "2023-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the video that corresponds to the following audio."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_shot2story(self, query_columns=["audio"], corpus_columns=["video"])


class Shot2Story20KVT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Shot2Story20KVT2ARetrieval",
        description=(
            "Retrieve the audio track that matches a given video clip and its summary "
            "caption from the Shot2Story20K benchmark of multi-shot videos."
        ),
        reference="https://arxiv.org/abs/2312.10300",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="vt2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text", "audio"],
        date=("2023-12-01", "2023-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Find the audio that corresponds to the following video and its description."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_shot2story(
            self, query_columns=["video", "caption"], corpus_columns=["audio"]
        )


class Shot2Story20KAT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Shot2Story20KAT2VRetrieval",
        description=(
            "Retrieve the video clip that matches a given audio track and its summary "
            "caption from the Shot2Story20K benchmark of multi-shot videos."
        ),
        reference="https://arxiv.org/abs/2312.10300",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="at2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "text", "video"],
        date=("2023-12-01", "2023-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Find the video that corresponds to the following audio and its description."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_shot2story(
            self, query_columns=["audio", "caption"], corpus_columns=["video"]
        )
