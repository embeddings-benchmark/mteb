from collections import defaultdict

import datasets
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

# Language codes for Common Voice 17
_EVAL_LANGS_CV17 = {
    "ab": ["abk-Latn"],  # Abkhaz
    "af": ["afr-Latn"],  # Afrikaans
    "am": ["amh-Ethi"],  # Amharic
    "ar": ["ara-Arab"],  # Arabic
    "as": ["asm-Beng"],  # Assamese
    "ast": ["ast-Latn"],  # Asturian
    "az": ["aze-Latn"],  # Azerbaijani
    "ba": ["bak-Cyrl"],  # Bashkir
    "bas": ["bas-Latn"],  # Basaa
    "be": ["bel-Cyrl"],  # Belarusian
    "bg": ["bul-Cyrl"],  # Bulgarian
    "bn": ["ben-Beng"],  # Bengali
    "br": ["bre-Latn"],  # Breton
    "ca": ["cat-Latn"],  # Catalan
    "ckb": ["ckb-Arab"],  # Central Kurdish (Sorani)
    "cnh": ["cnh-Latn"],  # Hakha Chin
    "cs": ["ces-Latn"],  # Czech
    "cv": ["chv-Cyrl"],  # Chuvash
    "cy": ["cym-Latn"],  # Welsh
    "da": ["dan-Latn"],  # Danish
    "de": ["deu-Latn"],  # German
    "dv": ["div-Thaa"],  # Divehi
    "dyu": ["dyu-Latn"],  # Dyula
    "el": ["ell-Grek"],  # Greek
    "en": ["eng-Latn"],  # English
    "eo": ["epo-Latn"],  # Esperanto
    "es": ["spa-Latn"],  # Spanish
    "et": ["est-Latn"],  # Estonian
    "eu": ["eus-Latn"],  # Basque
    "fa": ["fas-Arab"],  # Persian
    "fi": ["fin-Latn"],  # Finnish
    "fr": ["fra-Latn"],  # French
    "fy-NL": ["fry-Latn"],  # Frisian (Netherlands)
    "ga-IE": ["gle-Latn"],  # Irish (Ireland)
    "gl": ["glg-Latn"],  # Galician
    "gn": ["grn-Latn"],  # Guarani
    "ha": ["hau-Latn"],  # Hausa
    "he": ["heb-Hebr"],  # Hebrew
    "hi": ["hin-Deva"],  # Hindi
    "hsb": ["hsb-Latn"],  # Upper Sorbian
    "hu": ["hun-Latn"],  # Hungarian
    "hy-AM": ["hye-Armn"],  # Armenian (Armenia)
    "ia": ["ina-Latn"],  # Interlingua
    "id": ["ind-Latn"],  # Indonesian
    "ig": ["ibo-Latn"],  # Igbo
}

# Language codes for Common Voice 21
_EVAL_LANGS_CV21 = {
    "abk": ["abk-Latn"],  # Abkhaz
    "afr": ["afr-Latn"],  # Afrikaans
    "amh": ["amh-Ethi"],  # Amharic
    "ara": ["ara-Arab"],  # Arabic
    "asm": ["asm-Beng"],  # Assamese
    "ast": ["ast-Latn"],  # Asturian
    "aze": ["aze-Latn"],  # Azerbaijani
    "bak": ["bak-Cyrl"],  # Bashkir
    "bas": ["bas-Latn"],  # Basaa
    "bel": ["bel-Cyrl"],  # Belarusian
    "bul": ["bul-Cyrl"],  # Bulgarian
    "ben": ["ben-Beng"],  # Bengali
    "bre": ["bre-Latn"],  # Breton
    "cat": ["cat-Latn"],  # Catalan
    "ckb": ["ckb-Arab"],  # Central Kurdish (Sorani)
    "cnh": ["cnh-Latn"],  # Hakha Chin
    "ces": ["ces-Latn"],  # Czech
    "chv": ["chv-Cyrl"],  # Chuvash
    "cym": ["cym-Latn"],  # Welsh
    "dan": ["dan-Latn"],  # Danish
    "deu": ["deu-Latn"],  # German
    "div": ["div-Thaa"],  # Divehi
    "dyu": ["dyu-Latn"],  # Dyula
    "ell": ["ell-Grek"],  # Greek
    "eng": ["eng-Latn"],  # English
    "epo": ["epo-Latn"],  # Esperanto
    "spa": ["spa-Latn"],  # Spanish
    "est": ["est-Latn"],  # Estonian
    "eus": ["eus-Latn"],  # Basque
    "fas": ["fas-Arab"],  # Persian
    "fin": ["fin-Latn"],  # Finnish
    "fra": ["fra-Latn"],  # French
    "fry": ["fry-Latn"],  # Frisian (Netherlands)
    "gle": ["gle-Latn"],  # Irish (Ireland)
    "glg": ["glg-Latn"],  # Galician
    "grn": ["grn-Latn"],  # Guarani
    "hau": ["hau-Latn"],  # Hausa
    "heb": ["heb-Hebr"],  # Hebrew
    "hin": ["hin-Deva"],  # Hindi
    "hsb": ["hsb-Latn"],  # Upper Sorbian
    "hun": ["hun-Latn"],  # Hungarian
    "hye": ["hye-Armn"],  # Armenian (Armenia)
    "ina": ["ina-Latn"],  # Interlingua
    "ind": ["ind-Latn"],  # Indonesian
    "ibo": ["ibo-Latn"],  # Igbo
}


class CommonVoice17A2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice17A2TRetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "fsicoli/common_voice_17_0",
            "revision": "8262c16bf297c87a9cd88c51997c4758ed7a8ba2",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS_CV17,
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{ardila2019common,
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


class CommonVoice17T2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice17T2ARetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "fsicoli/common_voice_17_0",
            "revision": "8262c16bf297c87a9cd88c51997c4758ed7a8ba2",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS_CV17,
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{ardila2019common,
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


class CommonVoice21A2TRetrieval(AbsTaskRetrieval):
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
        eval_langs=_EVAL_LANGS_CV21,
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{ardila2019common,
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


class CommonVoice21T2ARetrieval(AbsTaskRetrieval):
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
        eval_langs=_EVAL_LANGS_CV21,
        main_score="cv_recall_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{ardila2019common,
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
