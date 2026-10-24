from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/OmnilingualASR-retrieval"
_DATASET_REVISION = "20df6efe1bdc72736c12faf99c5e6d365c176f35"

_LANGUAGES = {
    "aal": ["aal-Latn"],
    "abn": ["abn-Latn"],
    "abr": ["abr-Latn"],
    "abs": ["abs-Latn"],
    "aec": ["aec-Arab"],
    "afo": ["afo-Latn"],
    "ahl": ["ahl-Latn"],
    "ahs": ["ahs-Latn"],
    "ala": ["ala-Latn"],
    "alo": ["alo-Latn"],
    "amu": ["amu-Latn"],
    "anc": ["anc-Latn"],
    "ank": ["ank-Latn"],
    "anp": ["anp-Deva"],
    "anw": ["anw-Latn"],
    "aom": ["aom-Latn"],
    "apd": ["apd-Arab"],
    "ary": ["ary-Arab"],
    "awo": ["awo-Latn"],
    "ayl": ["ayl-Arab"],
    "ayp": ["ayp-Arab"],
    "bbu": ["bbu-Latn"],
    "bcs": ["bcs-Latn"],
    "bcy": ["bcy-Latn"],
    "bda": ["bda-Latn"],
    "bde": ["bde-Latn"],
    "bdm": ["bdm-Latn"],
    "bho": ["bho-Deva"],
    "bjj": ["bjj-Deva"],
    "bra": ["bra-Deva"],
    "brx": ["brx-Deva"],
    "dcc": ["dcc-Arab"],
    "dty": ["dty-Deva"],
    "gbm": ["gbm-Deva"],
    "gjk": ["gjk-Arab"],
    "gom": ["gom-Deva"],
    "kas": ["kas-Arab"],
    "knn": ["knn-Deva"],
    "kxp": ["kxp-Arab"],
    "mrr": ["mrr-Deva"],
    "mtr": ["mtr-Deva"],
    "odk": ["odk-Arab"],
    "phr": ["phr-Arab"],
    "pnb": ["pnb-Arab"],
    "sin": ["sin-Sinh"],
    "tcy": ["tcy-Mlym"],
    "the": ["the-Deva"],
    "thq": ["thq-Deva"],
    "tkt": ["tkt-Deva"],
    "uki": ["uki-Orya"],
}

_BIBTEX = r"""
@article{omnilingualasr2025,
  author = {{Omnilingual ASR Team}},
  journal = {arXiv preprint arXiv:2511.09690},
  title = {Omnilingual {ASR}: Open-Source Multilingual Speech Recognition for 1600+ Languages},
  year = {2025},
}
"""

_DESCRIPTION = (
    "Speech and its human transcription in 50 languages that no other audio task in "
    "mteb covers. Between them the existing audio tasks reach 165 languages, and none "
    "of these 50 is among them, so the point here is reach into low-resource languages "
    "rather than another benchmark on well-served ones."
)

_CONSTRUCTION = (
    "Built from the official test split. Languages are picked by a fixed rule: absent "
    "from every existing mteb audio task, test shard between 120MB and 320MB, then taken "
    "round-robin across writing systems so the selection spans seven scripts rather than "
    "one. Recordings are resampled to 16 kHz and re-encoded from FLAC to Opus, about "
    "eleven times smaller. Repeated transcripts are dropped, since one would otherwise "
    "be relevant to several recordings while only one is marked correct. Construction "
    "script: scripts/data/omnilingual_asr_retrieval/create_data.py."
)

_COMMON = {
    "reference": "https://arxiv.org/abs/2511.09690",
    "dataset": {"path": _DATASET_PATH, "revision": _DATASET_REVISION},
    "type": "Any2AnyMultilingualRetrieval",
    "eval_splits": ["test"],
    "eval_langs": _LANGUAGES,
    "main_score": "ndcg_at_10",
    "date": ("2024-01-01", "2025-11-12"),
    "domains": ["Spoken"],
    "task_subtypes": ["Cross-Modal Retrieval"],
    "license": "cc-by-4.0",
    "annotations_creators": "human-annotated",
    "dialect": [],
    "sample_creation": "created",
    "bibtex_citation": _BIBTEX,
    "is_beta": True,
}


def _load(task: AbsTaskRetrieval, to_text: bool) -> None:
    """Load one direction. `to_text` selects speech->transcript, else the reverse."""
    if task.data_loaded:
        return

    split = task.metadata.eval_splits[0]
    task.dataset = {}
    for lang in _LANGUAGES:
        rows = load_dataset(
            _DATASET_PATH, lang, revision=_DATASET_REVISION, split=split
        )
        audio = rows.select_columns(["id", "audio"])
        text = rows.select_columns(["id", "text"])
        ids = rows["id"]
        qrels = {i: {i: 1} for i in ids}

        task.dataset[lang] = {
            split: RetrievalSplitData(
                queries=audio if to_text else text,
                corpus=text if to_text else audio,
                relevant_docs=qrels,
                top_ranked=None,
            )
        }
    task.data_loaded = True


class OmnilingualASRA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OmnilingualASRA2TRetrieval",
        description=f"{_DESCRIPTION} Retrieve the transcription of a recording. {_CONSTRUCTION}",
        category="a2t",
        modalities=["audio", "text"],
        prompt={"query": "Find the transcription of this recording."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load(self, to_text=True)


class OmnilingualASRT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OmnilingualASRT2ARetrieval",
        description=f"{_DESCRIPTION} Retrieve the recording a transcription belongs to. {_CONSTRUCTION}",
        category="t2a",
        modalities=["text", "audio"],
        prompt={"query": "Find the recording of this transcription."},
        **_COMMON,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load(self, to_text=False)
