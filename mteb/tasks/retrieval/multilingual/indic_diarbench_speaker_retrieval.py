from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/IndicDiarBench-speaker-retrieval"
_DATASET_REVISION = "0e426b97c9a792fa6c10efc12b2d0800a1ff4257"

_LANGUAGES = {
    "asm": ["asm-Beng"],
    "ben": ["ben-Beng"],
    "brx": ["brx-Deva"],
    "doi": ["doi-Deva"],
    "guj": ["guj-Gujr"],
    "hin": ["hin-Deva"],
    "kan": ["kan-Knda"],
    "kas": ["kas-Arab"],
    "kok": ["gom-Deva"],
    "mal": ["mal-Mlym"],
    "mar": ["mar-Deva"],
    "mni": ["mni-Beng"],
    "npi": ["npi-Deva"],
    "ory": ["ory-Orya"],
    "pan": ["pan-Guru"],
    "san": ["san-Deva"],
    "sat": ["sat-Olck"],
    "tam": ["tam-Taml"],
    "tel": ["tel-Telu"],
    "urd": ["urd-Arab"],
}

_BIBTEX = r"""
@inproceedings{mehendale2026indicdiarbench,
  author = {Mehendale, Deovrat and Mehndiratta, Aditya and Rathi, Dhruv and Bhogale, Kaushal and Khapra, Mitesh M.},
  booktitle = {Interspeech},
  title = {{Indic DiarBench}: A Multilingual Joint Diarization and {ASR} Benchmark for {Indian} Languages},
  year = {2026},
}
"""


class IndicDiarBenchSpeakerA2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="IndicDiarBenchSpeakerA2ARetrieval",
        description=(
            "Speaker retrieval over conversational speech in 20 scheduled languages of "
            "India: given a clip of one speaker, find other clips of that same speaker. "
            "Speaker labels are numbered per recording session, so identity pairs the "
            "session with the speaker, and the corpus holds the other speakers of the "
            "same session as the hardest distractors: same room, same channel, different "
            "voice. Built from the official test split by cutting annotated turns to "
            "clips of 2 to 15 seconds, dropping turns that overlap a different speaker "
            "by more than 0.5s, and keeping speakers with at least five clips. Maithili "
            "and Sindhi are excluded because the source gives them only four speakers "
            "each. Construction script: "
            "scripts/data/indic_diarbench_speaker/create_data.py."
        ),
        reference="https://arxiv.org/abs/2607.23808",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyMultilingualRetrieval",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2025-01-01", "2026-07-26"),
        domains=["Spoken"],
        task_subtypes=["Speaker Identification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        prompt={"query": "Find other recordings of this speaker."},
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        split = self.metadata.eval_splits[0]
        self.dataset = {}
        for lang in _LANGUAGES:
            queries = load_dataset(
                _DATASET_PATH,
                f"{lang}-queries",
                revision=_DATASET_REVISION,
                split=split,
            )
            corpus = load_dataset(
                _DATASET_PATH, f"{lang}-corpus", revision=_DATASET_REVISION, split=split
            )

            # Read the identity columns directly; iterating full rows would decode audio.
            by_identity: dict[str, list[str]] = {}
            for doc_id, identity in zip(corpus["id"], corpus["identity"], strict=True):
                by_identity.setdefault(identity, []).append(doc_id)

            qrels = {
                qid: dict.fromkeys(by_identity.get(identity, []), 1)
                for qid, identity in zip(
                    queries["id"], queries["identity"], strict=True
                )
            }

            self.dataset[lang] = {
                split: RetrievalSplitData(
                    queries=queries.select_columns(["id", "audio"]),
                    corpus=corpus.select_columns(["id", "audio"]),
                    relevant_docs=qrels,
                    top_ranked=None,
                )
            }
        self.data_loaded = True
