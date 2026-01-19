from collections import defaultdict

import datasets
from datasets import Audio, DatasetDict

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
            "path": "google/svq",
            "revision": "177e4fa88e59148dc746471e164b0b46b193f41f",
            "name": "audio",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="cv_recall_at_5",
        date=("2024-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
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

    def load_data(self, **kwargs):
        if getattr(self, "data_loaded", False):
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self, id_col="utt_id", text_col="text", audio_col="waveform"):
        """A2T: Query = audio, Corpus = text."""
        for split in self.metadata.eval_splits:
            full_dataset = datasets.load_dataset(
                self.metadata.dataset["path"],
                self.metadata.dataset.get("name", "audio"),
                split=split,
                revision=self.metadata.dataset.get("revision"),
            )

            # Cast once before filtering to avoid multiple castings
            full_dataset = full_dataset.cast_column(audio_col, Audio(decode=True))

            for locale, _ in self.metadata.eval_langs.items():
                # Filter by locale
                lang_dataset = full_dataset.filter(lambda x: x["locale"] == locale)

                # Create datasets directly without intermediate lists
                queries_ds = (
                    lang_dataset.select_columns([id_col, audio_col])
                    .rename_column(id_col, "id")
                    .rename_column(audio_col, "audio")
                )

                corpus_ds = (
                    lang_dataset.select_columns([id_col, text_col])
                    .rename_column(id_col, "id")
                    .rename_column(text_col, "text")
                )

                # Create relevant_docs mapping
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
            "path": "google/svq",
            "revision": "177e4fa88e59148dc746471e164b0b46b193f41f",
            "name": "audio",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="cv_recall_at_5",
        date=("2024-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
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

    def load_data(self, **kwargs):
        if getattr(self, "data_loaded", False):
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self, id_col="utt_id", text_col="text", audio_col="waveform"):
        """T2A: Query = text, Corpus = audio."""
        for split in self.metadata.eval_splits:
            full_dataset = datasets.load_dataset(
                self.metadata.dataset["path"],
                self.metadata.dataset.get("name", "audio"),
                split=split,
                revision=self.metadata.dataset.get("revision"),
            )

            full_dataset = full_dataset.cast_column(audio_col, Audio(decode=True))

            for locale, _ in self.metadata.eval_langs.items():
                lang_dataset = full_dataset.filter(lambda x: x["locale"] == locale)

                queries_ds = (
                    lang_dataset.select_columns([id_col, text_col])
                    .rename_column(id_col, "id")
                    .rename_column(text_col, "text")
                )
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
