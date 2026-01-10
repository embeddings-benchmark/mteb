from collections import defaultdict

import datasets
from datasets import Audio, Dataset, DatasetDict
from tqdm import tqdm

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
        type="Any2AnyMultilingualRetrieval",
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

            for locale, _ in self.metadata.eval_langs.items():
                queries_ = {"id": [], "modality": [], "audio": []}
                corpus_ = {"id": [], "modality": [], "text": []}
                relevant_docs_ = {}

                # Filter BEFORE casting to Audio to avoid audio decoding during filter
                lang_dataset = full_dataset.filter(lambda x: x["locale"] == locale)
                # Cast waveform column to Audio type for proper decoding
                lang_dataset = lang_dataset.cast_column(audio_col, Audio(decode=True))

                qid = set()
                did = set()
                for row in tqdm(
                    lang_dataset,
                    total=len(lang_dataset),
                    desc=f"{locale}-{split}",
                ):
                    query_id = str(row[id_col])
                    doc_id = str(row[id_col])
                    text = row[text_col]
                    audio = row[audio_col]

                    if query_id not in qid:
                        qid.add(query_id)
                        queries_["id"].append(query_id)
                        queries_["audio"].append(audio)
                        queries_["modality"].append("audio")

                    if doc_id not in did:
                        did.add(doc_id)
                        corpus_["id"].append(doc_id)
                        corpus_["text"].append(text)
                        corpus_["modality"].append("text")

                    if query_id not in relevant_docs_:
                        relevant_docs_[query_id] = {}
                    relevant_docs_[query_id][doc_id] = 1

                corpus_ds = Dataset.from_dict(corpus_)
                queries_ds = Dataset.from_dict(queries_)
                # Cast the audio column to Audio type for proper encoding
                queries_ds = queries_ds.cast_column("audio", Audio(decode=True))

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
        type="Any2AnyMultilingualRetrieval",
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

            for locale, _ in self.metadata.eval_langs.items():
                queries_ = {"id": [], "modality": [], "text": []}
                corpus_ = {"id": [], "modality": [], "audio": []}
                relevant_docs_ = {}

                # Filter BEFORE casting to Audio to avoid audio decoding during filter
                lang_dataset = full_dataset.filter(lambda x: x["locale"] == locale)
                # Cast waveform column to Audio type for proper decoding
                lang_dataset = lang_dataset.cast_column(audio_col, Audio(decode=True))

                qid = set()
                did = set()
                for row in tqdm(
                    lang_dataset,
                    total=len(lang_dataset),
                    desc=f"{locale}-{split}",
                ):
                    query_id = str(row[id_col])
                    doc_id = str(row[id_col])
                    text = row[text_col]
                    audio = row[audio_col]

                    if query_id not in qid:
                        qid.add(query_id)
                        queries_["id"].append(query_id)
                        queries_["text"].append(text)
                        queries_["modality"].append("text")

                    if doc_id not in did:
                        did.add(doc_id)
                        corpus_["id"].append(doc_id)
                        corpus_["audio"].append(audio)
                        corpus_["modality"].append("audio")

                    if query_id not in relevant_docs_:
                        relevant_docs_[query_id] = {}
                    relevant_docs_[query_id][doc_id] = 1

                queries_ds = Dataset.from_dict(queries_)
                corpus_ds = Dataset.from_dict(corpus_)
                # Cast the audio column to Audio type for proper encoding
                corpus_ds = corpus_ds.cast_column("audio", Audio(decode=True))

                self.corpus[locale][split] = corpus_ds
                self.queries[locale][split] = queries_ds
                self.relevant_docs[locale][split] = relevant_docs_
