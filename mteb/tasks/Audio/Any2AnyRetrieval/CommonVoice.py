from __future__ import annotations

from collections import defaultdict

import datasets
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_LANGS = {
    "ab": ["Abkhazian-Latn"],
    "af": ["Afrikaans-Latn"],
    "am": ["Amharic-Ethi"],
    "ar": ["Arabic-Arab"],
    "as": ["Assamese-Beng"],
    "ast": ["Asturian-Latn"],
    "az": ["Azerbaijani-Latn"],
    "ba": ["Bashkir-Cyrl"],
    "bas": ["Basaa-Latn"],
    "be": ["Belarusian-Cyrl"],
    "bg": ["Bulgarian-Cyrl"],
    "bn": ["Bengali-Beng"],
    "br": ["Breton-Latn"],
    "ca": ["Catalan-Latn"],
    "ckb": ["Central Kurdish (Sorani)-Arab"],
    "cnh": ["Hakha Chin-Latn"],
    "cs": ["Czech-Latn"],
    "cv": ["Chuvash-Cyrl"],
    "cy": ["Welsh-Latn"],
    "da": ["Danish-Latn"],
    "de": ["German-Latn"],
    "dv": ["Divehi-Thaa"],
    "dyu": ["Dyula-Latn"],
    "el": ["Greek-Grek"],
    "en": ["English-Latn"],
    "eo": ["Esperanto-Latn"],
    "es": ["Spanish-Latn"],
    "et": ["Estonian-Latn"],
    "eu": ["Basque-Latn"],
    "fa": ["Persian-Arab"],
    "fi": ["Finnish-Latn"],
    "fr": ["French-Latn"],
    "fy-NL": ["Frisian (Netherlands)-Latn"],
    "ga-IE": ["Irish (Ireland)-Latn"],
    "gl": ["Galician-Latn"],
    "gn": ["Guarani-Latn"],
    "ha": ["Hausa-Latn"],
    "he": ["Hebrew-Hebr"],
    "hi": ["Hindi-Deva"],
    "hsb": ["Upper Sorbian-Latn"],
    "hu": ["Hungarian-Latn"],
    "hy-AM": ["Armenian (Armenia)-Armn"],
    "ia": ["Interlingua-Latn"],
    "id": ["Indonesian-Latn"],
    "ig": ["Igbo-Latn"],
}


class CommonVoice17A2TRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice17A2TRetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mozilla-foundation/common_voice_17_0",
            "revision": "b10d53980ef166bc24ce3358471c1970d7e6b5ec",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation = """@inproceedings{ardila2019common,
  author = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
  booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
  pages = {4218--4222},
  title = {Common voice: A massively-multilingual speech corpus},
  year = {2020},
}
"""
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self, id_col="path", text_col="sentence", audio_col="audio"):
        """Transform Common Voice dataset to MTEB t2a retrieval format.
        Returns (corpus, queries, relevant_docs) as DatasetDicts.
        """
        queries_ = {"id": [], "modality": [], "audio": []}
        corpus_ = {"id": [], "modality": [], "text": []}
        relevant_docs_ = {"query-id": [], "corpus-id": [], "score": []}
        relevant_docs_ = {}

        qid = set()
        did = set()
        for lang in self.metadata.eval_langs:
            lang_dataset = datasets.load_dataset(
                self.metadata.dataset["path"],
                lang,
                revision=self.metadata.dataset.get("revision"),
            )
            for split in self.metadata.eval_splits:
                for row in tqdm(lang_dataset[split], total=len(lang_dataset[split])):
                    # Use the "path" field as a unique id for both query and doc

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

                self.corpus[lang][split] = Dataset.from_dict(corpus_)
                self.queries[lang][split] = Dataset.from_dict(queries_)
                self.relevant_docs[lang][split] = (
                    relevant_docs_  # Dataset.from_dict(relevant_docs_)
                )


class CommonVoice17T2ARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice17T2ARetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mozilla-foundation/common_voice_17_0",
            "revision": "b10d53980ef166bc24ce3358471c1970d7e6b5ec",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation = """@inproceedings{ardila2019common,
  author = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
  booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
  pages = {4218--4222},
  title = {Common voice: A massively-multilingual speech corpus},
  year = {2020},
}
""",
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self, id_col="path", text_col="sentence", audio_col="audio"):
        """For T2A: query=text, corpus=audio."""
        queries_ = {"id": [], "modality": [], "text": []}
        corpus_ = {"id": [], "modality": [], "audio": []}
        relevant_docs_ = {}

        qid = set()
        did = set()
        for lang in self.metadata.eval_langs:
            lang_dataset = datasets.load_dataset(
                self.metadata.dataset["path"],
                lang,
                revision=self.metadata.dataset.get("revision"),
            )
            for split in self.metadata.eval_splits:
                for row in tqdm(lang_dataset[split], total=len(lang_dataset[split])):
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

                self.corpus[lang][split] = Dataset.from_dict(corpus_)
                self.queries[lang][split] = Dataset.from_dict(queries_)
                self.relevant_docs[lang][split] = relevant_docs_


class CommonVoice21A2TRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice21A2TRetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mteb/common_voice_21_0",
            "revision": "447fefbe174635d0f7073acd6503b3e84518dcea",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation = """@inproceedings{ardila2019common,
  author = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
  booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
  pages = {4218--4222},
  title = {Common voice: A massively-multilingual speech corpus},
  year = {2020},
}
"""
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self, id_col="path", text_col="sentence", audio_col="audio"):
        """Transform Common Voice dataset to MTEB t2a retrieval format.
        Returns (corpus, queries, relevant_docs) as DatasetDicts.
        """
        queries_ = {"id": [], "modality": [], "audio": []}
        corpus_ = {"id": [], "modality": [], "text": []}
        relevant_docs_ = {"query-id": [], "corpus-id": [], "score": []}
        relevant_docs_ = {}

        qid = set()
        did = set()
        for lang in self.metadata.eval_langs:
            lang_dataset = datasets.load_dataset(
                self.metadata.dataset["path"],
                lang,
                revision=self.metadata.dataset.get("revision"),
            )
            for split in self.metadata.eval_splits:
                for row in tqdm(lang_dataset[split], total=len(lang_dataset[split])):
                    # Use the "path" field as a unique id for both query and doc

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

                self.corpus[lang][split] = Dataset.from_dict(corpus_)
                self.queries[lang][split] = Dataset.from_dict(queries_)
                self.relevant_docs[lang][split] = (
                    relevant_docs_  # Dataset.from_dict(relevant_docs_)
                )


class CommonVoice21T2ARetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice21T2ARetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mteb/common_voice_21_0",
            "revision": "447fefbe174635d0f7073acd6503b3e84518dcea",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation = """@inproceedings{ardila2019common,
  author = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Henretty, Michael and Kohler, Michael and Meyer, Josh and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M and Weber, Gregor},
  booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
  pages = {4218--4222},
  title = {Common voice: A massively-multilingual speech corpus},
  year = {2020},
}
"""

    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self, id_col="path", text_col="sentence", audio_col="audio"):
        """For T2A: query=text, corpus=audio."""
        queries_ = {"id": [], "modality": [], "text": []}
        corpus_ = {"id": [], "modality": [], "audio": []}
        relevant_docs_ = {}

        qid = set()
        did = set()
        for lang in self.metadata.eval_langs:
            lang_dataset = datasets.load_dataset(
                self.metadata.dataset["path"],
                lang,
                revision=self.metadata.dataset.get("revision"),
            )
            for split in self.metadata.eval_splits:
                for row in tqdm(lang_dataset[split], total=len(lang_dataset[split])):
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

                self.corpus[lang][split] = Dataset.from_dict(corpus_)
                self.queries[lang][split] = Dataset.from_dict(queries_)
                self.relevant_docs[lang][split] = relevant_docs_
