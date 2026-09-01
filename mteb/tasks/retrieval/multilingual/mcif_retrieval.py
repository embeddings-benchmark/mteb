from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/MCIF-retrieval"
_DATASET_REVISION = "112f5bb92217b25845cf3fc28be529bc1439cc1f"

_LANGUAGES = {
    "en": ["eng-Latn"],
    "de": ["deu-Latn"],
    "it": ["ita-Latn"],
    "zh": ["cmn-Hans"],
}

_BIBTEX = r"""
@article{papi2025mcif,
  author = {Papi, Sara and Z{\"u}fle, Maike and Gaido, Marco and Savoldi, Beatrice and Liu, Danni and Douros, Ioannis and Bentivogli, Luisa and Niehues, Jan},
  journal = {arXiv preprint arXiv:2507.19634},
  title = {{MCIF}: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks},
  year = {2025},
}
"""

_DESCRIPTION = (
    "Retrieval over recorded scientific talks from MCIF. The talks are delivered in "
    "English while the questions are parallel across English, German, Italian and "
    "Chinese, so the three non-English subsets score cross-lingual grounding rather "
    "than same-language matching."
)

_CONSTRUCTION = (
    "Built from MCIF's question-answering samples, the only entries binding one question "
    "to one clip; its recognition and translation samples group roughly 33 clips behind a "
    "single reference. Unanswerable questions and ones about the speaker or affiliation "
    "are dropped as they match many talks, leaving 133 content-specific questions per "
    "language against all 755 clips. Construction script: "
    "scripts/data/mcif_retrieval/create_data.py."
)

_COMMON = {
    "reference": "https://arxiv.org/abs/2507.19634",
    "dataset": {"path": _DATASET_PATH, "revision": _DATASET_REVISION},
    "type": "Any2AnyMultilingualRetrieval",
    "eval_splits": ["test"],
    "eval_langs": _LANGUAGES,
    "main_score": "ndcg_at_10",
    "date": ("2023-07-01", "2025-07-25"),
    "domains": ["Academic", "Spoken"],
    "task_subtypes": ["Cross-Modal Retrieval"],
    "license": "cc-by-4.0",
    "annotations_creators": "human-annotated",
    "dialect": [],
    "sample_creation": "created",
    "bibtex_citation": _BIBTEX,
    "is_beta": True,
}


def _load_mcif(task: AbsTaskRetrieval, media_col: str, to_text: bool) -> None:
    """Load one MCIF direction across the four question languages.

    `to_text` selects media->question; otherwise question->media. Only the requested
    media column is exposed, so the video tasks cannot be answered from the audio track.
    """
    if task.data_loaded:
        return

    split = task.metadata.eval_splits[0]
    media = load_dataset(
        _DATASET_PATH, "media", revision=_DATASET_REVISION, split=split
    ).select_columns(["id", media_col])
    questions = load_dataset(
        _DATASET_PATH, "questions", revision=_DATASET_REVISION, split=split
    )

    # Read the link columns directly; iterating full rows would decode every clip.
    links = list(zip(questions["id"], questions["media_id"], strict=True))
    # Every question points at a clip, so the queried subset is the same in all languages.
    asked = {mid for _, mid in links}
    wanted = [i for i, id_ in enumerate(media["id"]) if id_ in asked]

    task.dataset = {}
    for lang in _LANGUAGES:
        text_ds = questions.select_columns(["id", f"text_{lang}"]).rename_column(
            f"text_{lang}", "text"
        )
        if to_text:
            qrels: dict[str, dict[str, int]] = {}
            for qid, mid in links:
                qrels.setdefault(mid, {})[qid] = 1
            # select() by index rather than filter(), which would decode every clip
            queries, corpus = media.select(wanted), text_ds
        else:
            queries, corpus = text_ds, media
            qrels = {qid: {mid: 1} for qid, mid in links}

        task.dataset[lang] = {
            split: RetrievalSplitData(
                queries=queries, corpus=corpus, relevant_docs=qrels, top_ranked=None
            )
        }
    task.data_loaded = True


class MCIFT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MCIFT2VRetrieval",
        description=f"{_DESCRIPTION} Retrieve the talk segment answering a question. {_CONSTRUCTION}",
        category="t2v",
        modalities=["text", "video"],
        prompt={"query": "Find the talk segment that answers this question."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_mcif(self, "video", to_text=False)


class MCIFV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MCIFV2TRetrieval",
        description=f"{_DESCRIPTION} Retrieve the question a talk segment answers. {_CONSTRUCTION}",
        category="v2t",
        modalities=["video", "text"],
        prompt={"query": "Find the question that this talk segment answers."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_mcif(self, "video", to_text=True)


class MCIFT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MCIFT2ARetrieval",
        description=f"{_DESCRIPTION} Retrieve the talk audio answering a question. {_CONSTRUCTION}",
        category="t2a",
        modalities=["text", "audio"],
        prompt={"query": "Find the talk audio that answers this question."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_mcif(self, "audio", to_text=False)


class MCIFA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MCIFA2TRetrieval",
        description=f"{_DESCRIPTION} Retrieve the question a talk audio answers. {_CONSTRUCTION}",
        category="a2t",
        modalities=["audio", "text"],
        prompt={"query": "Find the question that this talk audio answers."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_mcif(self, "audio", to_text=True)
