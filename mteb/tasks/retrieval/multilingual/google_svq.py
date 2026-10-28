import unicodedata
from collections import defaultdict
from typing import Any

import datasets
from datasets import Audio, DatasetDict
from tqdm.auto import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_LANGS = {
    "ar_eg": ["arz-Arab"],  # Egyptian Arabic
    "ar_x_gulf": ["acm-Arab"],  # Gulf Arabic
    "ar_x_levant": ["apc-Arab"],  # Levantine Arabic
    "ar_x_maghrebi": ["arq-Arab"],  # Maghrebi Arabic
    "bn_bd": ["ben-Beng"],  # Bengali (Bangladesh)
    "bn_in": ["ben-Beng"],  # Bengali (India)
    "en_au": ["eng-Latn"],  # English (Australia)
    "en_gb": ["eng-Latn"],  # English (GB)
    "en_in": ["eng-Latn"],  # English (India)
    "en_ph": ["eng-Latn"],  # English (Philippines)
    "en_us": ["eng-Latn"],  # English (US)
    "fi_fi": ["fin-Latn"],  # Finnish
    "gu_in": ["guj-Gujr"],  # Gujarati (India)
    "hi_in": ["hin-Deva"],  # Hindi (India)
    "id_id": ["ind-Latn"],  # Indonesian
    "ja_jp": ["jpn-Jpan"],  # Japanese
    "kn_in": ["kan-Knda"],  # Kannada (India)
    "ko_kr": ["kor-Hang"],  # Korean
    "ml_in": ["mal-Mlym"],  # Malayalam (India)
    "mr_in": ["mar-Deva"],  # Marathi (India)
    "ru_ru": ["rus-Cyrl"],  # Russian
    "sw": ["swa-Latn"],  # Swahili
    "ta_in": ["tam-Taml"],  # Tamil (India)
    "te_in": ["tel-Telu"],  # Telugu (India)
    "ur_in": ["urd-Arab"],  # Urdu (India)
    "ur_pk": ["urd-Arab"],  # Urdu (Pakistan)
}


class GoogleSVQA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GoogleSVQA2TRetrieval",
        description="Multilingual audio-to-text retrieval using the Simple Voice Questions (SVQ) dataset. Given an audio query, retrieve the corresponding text transcription.",
        reference="https://huggingface.co/datasets/google/svq",
        dataset={
            "path": "mteb/svq",
            "revision": "dd5a15e925767554c71b867f811b7eab53c770a4",
            "name": "audio",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="hit_rate_at_5",
        date=("2024-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        superseded_by="GoogleSVQA2TRetrieval.v2",
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{heigold2025massive,
  author = {Georg Heigold and Ehsan Variani and Tom Bagby and Cyril Allauzen and Ji Ma and Shankar Kumar and Michael Riley},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  title = {Massive Sound Embedding Benchmark ({MSEB})},
  url = {https://openreview.net/forum?id=X0juYgFVng},
  year = {2025},
}
""",
    )

    def load_data(self, **kwargs: Any):
        if getattr(self, "data_loaded", False):
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(
        self,
        id_col: str = "utt_id",
        text_col: str = "text",
        audio_col: str = "waveform",
    ) -> None:
        """A2T: Query = audio, Corpus = text."""
        for split in self.metadata.eval_splits:
            full_dataset = datasets.load_dataset(
                self.metadata.dataset["path"],
                self.metadata.dataset.get("name", "audio"),
                split=split,
                revision=self.metadata.dataset.get("revision"),
            )

            # Cast once before filtering
            full_dataset = full_dataset.cast_column(audio_col, Audio(decode=True))

            # Group indices by locale in a single pass
            locale_indices = {locale: [] for locale in self.metadata.eval_langs}

            for idx, row in enumerate(tqdm(full_dataset, desc="Filtering")):
                locale = row["locale"]
                if locale in locale_indices:
                    locale_indices[locale].append(idx)

            # Use select with indices to avoid recreating dataset
            for locale, indices in tqdm(
                locale_indices.items(), total=len(locale_indices)
            ):
                if not indices:
                    continue

                lang_dataset = full_dataset.select(indices)

                queries_ds = (
                    lang_dataset.select_columns([id_col, audio_col])
                    .rename_column(id_col, "id")
                    .rename_column(audio_col, "audio")
                )

                corpus_ds = lang_dataset.select_columns(
                    [id_col, text_col]
                ).rename_column(id_col, "id")

                relevant_docs_ = {
                    str(row[id_col]): {str(row[id_col]): 1} for row in lang_dataset
                }

                self.corpus[locale][split] = corpus_ds
                self.queries[locale][split] = queries_ds
                self.relevant_docs[locale][split] = relevant_docs_


class GoogleSVQT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GoogleSVQT2ARetrieval",
        description="Multilingual text-to-audio retrieval using the Simple Voice Questions (SVQ) dataset. Given a text query, retrieve the corresponding audio recording.",
        reference="https://huggingface.co/datasets/google/svq",
        dataset={
            "path": "mteb/svq",
            "revision": "dd5a15e925767554c71b867f811b7eab53c770a4",
            "name": "audio",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="hit_rate_at_5",
        date=("2024-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        superseded_by="GoogleSVQT2ARetrieval.v2",
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{heigold2025massive,
  author = {Georg Heigold and Ehsan Variani and Tom Bagby and Cyril Allauzen and Ji Ma and Shankar Kumar and Michael Riley},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  title = {Massive Sound Embedding Benchmark ({MSEB})},
  url = {https://openreview.net/forum?id=X0juYgFVng},
  year = {2025},
}
""",
    )

    def load_data(self, **kwargs: Any):
        if getattr(self, "data_loaded", False):
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(
        self,
        id_col: str = "utt_id",
        text_col: str = "text",
        audio_col: str = "waveform",
    ) -> None:
        """T2A: Query = text, Corpus = audio."""
        for split in self.metadata.eval_splits:
            full_dataset = datasets.load_dataset(
                self.metadata.dataset["path"],
                self.metadata.dataset.get("name", "audio"),
                split=split,
                revision=self.metadata.dataset.get("revision"),
            )

            full_dataset = full_dataset.cast_column(audio_col, Audio(decode=True))

            locale_indices = {locale: [] for locale in self.metadata.eval_langs}

            for idx, row in enumerate(tqdm(full_dataset, desc="Filtering")):
                locale = row["locale"]
                if locale in locale_indices:
                    locale_indices[locale].append(idx)

            # Use select with indices to avoid recreating dataset
            for locale, indices in tqdm(
                locale_indices.items(), total=len(locale_indices)
            ):
                if not indices:
                    continue

                lang_dataset = full_dataset.select(indices)

                queries_ds = lang_dataset.select_columns(
                    [id_col, text_col]
                ).rename_column(id_col, "id")

                corpus_ds = (
                    lang_dataset.select_columns([id_col, audio_col])
                    .rename_column(id_col, "id")
                    .rename_column(audio_col, "audio")
                )

                relevant_docs_ = {
                    str(row[id_col]): {str(row[id_col]): 1} for row in lang_dataset
                }

                self.corpus[locale][split] = corpus_ds
                self.queries[locale][split] = queries_ds
                self.relevant_docs[locale][split] = relevant_docs_


# Arabic diacritics and tatweel, and hamza forms of alef
# Arabic letter variants that are written interchangeably (hamza/alef forms, alef maksura, ta marbuta, keheh, tatweel)
_ARABIC_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ی": "ي",
        "ة": "ه",
        "ک": "ك",
        "ـ": "",
    }
)


def _normalize(text: str) -> str:
    """Fold transcription variants of the same sentence.

    Drops case, punctuation, symbols and extra whitespace, and folds Arabic letter variants,
    Arabic diacritics and accents on Latin letters. Indic vowel signs are kept, since they
    are combining marks that change the word.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(
        c for c in text if not unicodedata.category(c).startswith(("P", "S"))
    )
    out = []
    for ch in text.translate(_ARABIC_MAP):
        if unicodedata.name(ch, "").startswith("LATIN"):
            out.extend(
                d
                for d in unicodedata.normalize("NFD", ch)
                if not unicodedata.combining(d)
            )
        elif not (unicodedata.combining(ch) and "ARABIC" in unicodedata.name(ch, "")):
            out.append(ch)
    return " ".join("".join(out).split())


def _sentence_ids(utt_ids: list[str], texts: list[str]) -> list[str]:
    """Sentence id of each recording: the first recording id with the same normalized text."""
    first: dict[str, str] = {}
    return [
        first.setdefault(_normalize(text), utt_id)
        for utt_id, text in zip(utt_ids, texts, strict=True)
    ]


class GoogleSVQA2TRetrievalV2(AbsTaskRetrieval):
    metadata = GoogleSVQA2TRetrieval.metadata.model_copy(
        update={
            "name": "GoogleSVQA2TRetrieval.v2",
            "description": GoogleSVQA2TRetrieval.metadata.description
            + " Version 2 keeps one text per unique sentence: each sentence is recorded several times, and v1 listed every recording's identical transcription as a separate document.",
            "adapted_from": ["GoogleSVQA2TRetrieval"],
            "superseded_by": None,
        }
    )

    def load_data(self, **kwargs: Any) -> None:
        if getattr(self, "data_loaded", False):
            return
        # same rows as v1, then one entry per unique sentence
        v1 = GoogleSVQA2TRetrieval()
        v1.load_data()
        self.corpus, self.queries = v1.corpus, v1.queries
        self.relevant_docs = v1.relevant_docs
        for locale, splits in self.corpus.items():
            for split, corpus_ds in splits.items():
                utt_ids, texts = corpus_ds["id"], corpus_ds["text"]
                sentence_ids = _sentence_ids(utt_ids, texts)
                unique = dict(zip(sentence_ids, texts, strict=True))
                self.corpus[locale][split] = datasets.Dataset.from_dict(
                    {"id": list(unique), "text": list(unique.values())}
                )
                self.relevant_docs[locale][split] = {
                    utt_id: {sentence_id: 1}
                    for utt_id, sentence_id in zip(utt_ids, sentence_ids, strict=True)
                }
        self.data_loaded = True


class GoogleSVQT2ARetrievalV2(AbsTaskRetrieval):
    metadata = GoogleSVQT2ARetrieval.metadata.model_copy(
        update={
            "name": "GoogleSVQT2ARetrieval.v2",
            "description": GoogleSVQT2ARetrieval.metadata.description
            + " Version 2 has one query per unique sentence with all its recordings as correct answers: each sentence is recorded several times, and v1 asked the same text once per recording with only that recording correct.",
            "adapted_from": ["GoogleSVQT2ARetrieval"],
            "superseded_by": None,
        }
    )

    def load_data(self, **kwargs: Any) -> None:
        if getattr(self, "data_loaded", False):
            return
        # same rows as v1, then one entry per unique sentence
        v1 = GoogleSVQT2ARetrieval()
        v1.load_data()
        self.corpus, self.queries = v1.corpus, v1.queries
        self.relevant_docs = v1.relevant_docs
        for locale, splits in self.queries.items():
            for split, queries_ds in splits.items():
                utt_ids, texts = queries_ds["id"], queries_ds["text"]
                sentence_ids = _sentence_ids(utt_ids, texts)
                unique = dict(zip(sentence_ids, texts, strict=True))
                self.queries[locale][split] = datasets.Dataset.from_dict(
                    {"id": list(unique), "text": list(unique.values())}
                )
                relevant: dict[str, dict[str, int]] = defaultdict(dict)
                for utt_id, sentence_id in zip(utt_ids, sentence_ids, strict=True):
                    relevant[sentence_id][utt_id] = 1
                self.relevant_docs[locale][split] = dict(relevant)
        self.data_loaded = True
