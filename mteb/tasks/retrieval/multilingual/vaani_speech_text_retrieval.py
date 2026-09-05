from __future__ import annotations

from typing import Any

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/vaani-speech-text-retrieval"
_DATASET_REVISION = "fb93667a13461cd1652cd3780ecc59503fccdb11"

# Keys are Vaani's own language directories. Scripts were determined from the
# transcripts themselves rather than assumed: Chakma is romanised here despite having
# its own script, and Tulu is written in Kannada. Codes marked "approx" name a variety
# with no distinct ISO 639-3 entry and are mapped to the closest coded one.
_VAANI_ST_LANGS = {
    "Angika": ["anp-Deva"],
    "Assamese": ["asm-Beng"],
    "Bajjika": ["mai-Deva"],  # approx: no distinct ISO 639-3 entry
    "Bengali": ["ben-Beng"],
    "Bhojpuri": ["bho-Deva"],
    "Bundeli": ["bns-Deva"],
    "Chakma": ["ccp-Latn"],  # romanised in this release, not the Chakma script
    "Chhattisgarhi": ["hne-Deva"],
    "English": ["eng-Latn"],
    "Garhwali": ["gbm-Deva"],
    "Garo": ["grt-Latn"],
    "Gujarati": ["guj-Gujr"],
    "Halbi": ["hlb-Deva"],
    "Hindi": ["hin-Deva"],
    "IduMishmi": ["clk-Latn"],
    "Kannada": ["kan-Knda"],
    "Karbi": ["mjw-Latn"],
    "Kashmiri": ["kas-Arab"],
    "Khariboli": ["hin-Deva"],  # approx: the base dialect of Standard Hindi
    "Khortha": ["mag-Deva"],  # approx: no distinct ISO 639-3 entry
    "Kokborok": ["trp-Latn"],
    "Konkani": ["kok-Deva"],
    "Kumaoni": ["kfy-Deva"],
    "Magadhi": ["mag-Deva"],
    "Magahi": ["mag-Deva"],
    "Maithili": ["mai-Deva"],
    "Malayalam": ["mal-Mlym"],
    "Malvani": ["kok-Deva"],  # approx: no distinct ISO 639-3 entry
    "Marathi": ["mar-Deva"],
    "Marwari": ["mwr-Deva"],
    "Mizo": ["lus-Latn"],
    "Nagamese": ["nag-Latn"],
    "Nepali": ["npi-Deva"],
    "Odia": ["ory-Orya"],
    "Punjabi": ["pan-Guru"],
    "Rajasthani": ["raj-Deva"],
    "Rengma": ["nre-Latn"],
    "Sadri": ["sck-Deva"],
    "Sumi": ["nsm-Latn"],
    "Surgujia": ["sgj-Deva"],
    "Tamil": ["tam-Taml"],
    "Telugu": ["tel-Telu"],
    "Tulu": ["tcy-Knda"],  # Tulu is written in the Kannada script
    "Urdu": ["urd-Arab"],
    "Wancho": ["nnp-Latn"],
}

_BIBTEX = r"""
@misc{pulikodan2026vaani,
  archiveprefix = {arXiv},
  author = {Pulikodan, Sujith and Singh, Abhayjeet and Basu, Agneedh and Desai, Nihar and J, Pavan Kumar and Bhat, Pranav D and Dharmaraju, Raghu and Gupta, Ritika and Udupa, Sathvik and Kumar, Saurabh and Sharma, Sumit and Sanka, Visruth and Tewari, Dinesh and Dhand, Harsh and Kamat, Amrita and Singh, Sukhwinder and Vashishth, Shikhar and Talukdar, Partha and Acharya, Raj and Ghosh, Prasanta Kumar},
  eprint = {2603.28714},
  title = {VAANI: Capturing the language landscape for an inclusive digital India},
  year = {2026},
}
"""

_DESCRIPTION = (
    "Multilingual speech-transcript retrieval over 45 Indian languages, from Vaani's "
    "transcribed release. Unlike the main Vaani corpus this release ships an official "
    "test split, so the evaluation set is held out at source rather than sampled from "
    "training data. Each language is scored against its own transcript pool."
)


def _load_vaani_st(task: AbsTaskRetrieval, direction: str) -> None:
    """Shared loader for both directions."""
    if task.data_loaded:
        return

    split = task.metadata.eval_splits[0]
    task.dataset = {}

    for lang in task.hf_subsets:
        ds = load_dataset(_DATASET_PATH, lang, revision=_DATASET_REVISION, split=split)
        audio_ds = ds.select_columns(["id", "audio"])
        text_ds = ds.select_columns(["id", "text"])
        queries, corpus = (
            (audio_ds, text_ds) if direction == "a2t" else (text_ds, audio_ds)
        )

        task.dataset[lang] = {
            split: RetrievalSplitData(
                queries=queries,
                corpus=corpus,
                relevant_docs={i: {i: 1} for i in ds["id"]},
                top_ranked=None,
            )
        }

    task.data_loaded = True


class VaaniA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VaaniA2TRetrieval",
        description=f"{_DESCRIPTION} Retrieve the transcription of a spoken utterance.",
        reference="https://arxiv.org/abs/2603.28714",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="a2t",
        eval_splits=["test"],
        eval_langs=_VAANI_ST_LANGS,
        main_score="hit_rate_at_5",
        modalities=["audio", "text"],
        date=("2023-01-01", "2025-06-30"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the transcription of this spoken utterance."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_vaani_st(self, "a2t")


class VaaniT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VaaniT2ARetrieval",
        description=f"{_DESCRIPTION} Retrieve the recording matching a transcription.",
        reference="https://arxiv.org/abs/2603.28714",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2a",
        eval_splits=["test"],
        eval_langs=_VAANI_ST_LANGS,
        main_score="hit_rate_at_5",
        modalities=["text", "audio"],
        date=("2023-01-01", "2025-06-30"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the recording of the following transcription."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _load_vaani_st(self, "t2a")
