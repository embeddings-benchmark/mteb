from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "mteb/AVMeme-Exam"
_DATASET_REVISION = "7070d1979d9a4943dd49b2e72858eb1e54f6bd5b"
_BIBTEX = r"""
@article{avmeme2025,
  title = {AVMeme: A Benchmark for Audio-Visual Meme Understanding},
  year = {2025},
}
"""


def _load_avmeme_exam(
    task: AbsTaskRetrieval,
    query_columns: list[str],
    corpus_columns: list[str],
) -> None:
    """Shared loader for all AVMeme-Exam retrieval directions.

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
    if "summary" in query_columns:
        query = query.rename_column("summary", "text")
    if "summary" in corpus_columns:
        corpus = corpus.rename_column("summary", "text")

    qrels = {str(i): {str(i): 1} for i in range(len(dataset))}
    task.dataset["default"]["test"] = RetrievalSplitData(
        queries=query, corpus=corpus, relevant_docs=qrels, top_ranked=None
    )
    task.data_loaded = True


class AVMemeExamV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVMemeExamV2TRetrieval",
        description=(
            "Retrieve the summary that describes a given meme video clip (video "
            "only) from AVMeme-Exam, a multilingual audio-visual meme understanding "
            "benchmark."
        ),
        reference="https://huggingface.co/datasets/mteb/AVMeme-Exam",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2025-01-01", "2025-12-31"),
        domains=["Web", "Social"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the summary that describes the following meme video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_avmeme_exam(self, query_columns=["video"], corpus_columns=["summary"])


class AVMemeExamT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVMemeExamT2VRetrieval",
        description=(
            "Retrieve the meme video clip (video only) that matches a given summary "
            "from AVMeme-Exam, a multilingual audio-visual meme understanding "
            "benchmark."
        ),
        reference="https://huggingface.co/datasets/mteb/AVMeme-Exam",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2025-01-01", "2025-12-31"),
        domains=["Web", "Social"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the meme video clip that matches the given summary."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_avmeme_exam(self, query_columns=["summary"], corpus_columns=["video"])


class AVMemeExamVA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVMemeExamVA2TRetrieval",
        description=(
            "Retrieve the summary that describes a given meme video+audio clip "
            "from AVMeme-Exam, a multilingual audio-visual meme understanding "
            "benchmark."
        ),
        reference="https://huggingface.co/datasets/mteb/AVMeme-Exam",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="va2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video", "text"],
        date=("2025-01-01", "2025-12-31"),
        domains=["Web", "Social"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the summary that describes the following meme video."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_avmeme_exam(
            self, query_columns=["video", "audio"], corpus_columns=["summary"]
        )


class AVMemeExamT2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVMemeExamT2VARetrieval",
        description=(
            "Retrieve the meme video+audio clip that matches a given summary "
            "from AVMeme-Exam, a multilingual audio-visual meme understanding "
            "benchmark."
        ),
        reference="https://huggingface.co/datasets/mteb/AVMeme-Exam",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2va",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["text", "audio", "video"],
        date=("2025-01-01", "2025-12-31"),
        domains=["Web", "Social"],
        task_subtypes=["Caption Pairing"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the meme video clip that matches the given summary."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_avmeme_exam(
            self, query_columns=["summary"], corpus_columns=["video", "audio"]
        )


class AVMemeExamV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVMemeExamV2ARetrieval",
        description=(
            "Retrieve the audio track that matches a given meme video clip (video "
            "only) from AVMeme-Exam. Tests cross-modal alignment between meme visual "
            "content and its paired audio."
        ),
        reference="https://huggingface.co/datasets/mteb/AVMeme-Exam",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="v2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "audio"],
        date=("2025-01-01", "2025-12-31"),
        domains=["Web", "Social"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Find the audio that corresponds to the following meme video."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_avmeme_exam(self, query_columns=["video"], corpus_columns=["audio"])


class AVMemeExamA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AVMemeExamA2VRetrieval",
        description=(
            "Retrieve the meme video clip that matches a given audio track from "
            "AVMeme-Exam. Tests cross-modal alignment between meme audio and its "
            "paired visual content."
        ),
        reference="https://huggingface.co/datasets/mteb/AVMeme-Exam",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="a2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["audio", "video"],
        date=("2025-01-01", "2025-12-31"),
        domains=["Web", "Social"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Find the meme video that corresponds to the following audio."
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_avmeme_exam(self, query_columns=["audio"], corpus_columns=["video"])
