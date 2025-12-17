from collections import defaultdict

import datasets
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

# Language codes for Common Voice 17
_EVAL_LANGS_CV17 = {
    "ar": ["ara-Arab"],
    "ast": ["ast-Latn"],
    "be": ["bel-Cyrl"],
    "bg": ["bul-Cyrl"],
    "bn": ["ben-Beng"],
    "br": ["bre-Latn"],
    "cs": ["ces-Latn"],
    "cy": ["cym-Latn"],
    "da": ["dan-Latn"],
    "de": ["deu-Latn"],
    "el": ["ell-Grek"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "et": ["est-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "fro": ["fro-Latn"],
    "gl": ["glg-Latn"],
    "ha": ["hau-Latn"],
    "hi": ["hin-Deva"],
    "hu": ["hun-Latn"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "ka": ["kat-Geor"],
    "ko": ["kor-Hang"],
    "lt": ["lit-Latn"],
    "lv": ["lav-Latn"],
    "mk": ["mkd-Cyrl"],
    "ml": ["mal-Mlym"],
    "mn": ["mon-Cyrl"],
    "mr": ["mar-Deva"],
    "nl": ["nld-Latn"],
    "oc": ["oci-Latn"],
    "pl": ["pol-Latn"],
    "pt": ["por-Latn"],
    "ro": ["ron-Latn"],
    "ru": ["rus-Cyrl"],
    "sk": ["slk-Latn"],
    "sl": ["slv-Latn"],
    "sr": ["srp-Cyrl"],
    "sv-SE": ["swe-Latn"],
    "sw": ["swa-Latn"],
    "ta": ["tam-Taml"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "tr": ["tur-Latn"],
    "uk": ["ukr-Cyrl"],
    "ur": ["urd-Arab"],
    "vi": ["vie-Latn"],
}

# Language codes for Common Voice 21
_EVAL_LANGS_CV21 = {
    "ab": ["abk-Cyrl"],
    "af": ["afr-Latn"],
    "am": ["amh-Ethi"],
    "ar": ["ara-Arab"],
    "as": ["asm-Beng"],
    "ast": ["ast-Latn"],
    "az": ["aze-Latn"],
    "bas": ["bas-Latn"],
    "be": ["bel-Cyrl"],
    "bg": ["bul-Cyrl"],
    "br": ["bre-Latn"],
    "ckb": ["ckb-Arab"],
    "cnh": ["cnh-Latn"],
    "cs": ["ces-Latn"],
    "cv": ["chv-Cyrl"],
    "cy": ["cym-Latn"],
    "da": ["dan-Latn"],
    "dav": ["dav-Latn"],
    "de": ["deu-Latn"],
    "dv": ["div-Thaa"],
    "dyu": ["dyu-Latn"],
    "el": ["ell-Grek"],
    "en": ["eng-Latn"],
    "eo": ["epo-Latn"],
    "es": ["spa-Latn"],
    "et": ["est-Latn"],
    "eu": ["eus-Latn"],
    "fr": ["fra-Latn"],
    "fy-NL": ["fry-Latn"],
    "ga-IE": ["gle-Latn"],
    "gl": ["glg-Latn"],
    "he": ["heb-Hebr"],
    "hi": ["hin-Deva"],
    "hsb": ["hsb-Latn"],
    "ht": ["hat-Latn"],
    "hu": ["hun-Latn"],
    "hy-AM": ["hye-Armn"],
    "id": ["ind-Latn"],
    "ig": ["ibo-Latn"],
    "is": ["isl-Latn"],
    "it": ["ita-Latn"],
    "kk": ["kaz-Cyrl"],
    "kln": ["kln-Latn"],
    "kmr": ["kmr-Latn"],
    "ko": ["kor-Hang"],
    "ky": ["kir-Cyrl"],
    "lij": ["lij-Latn"],
    "lo": ["lao-Laoo"],
    "lt": ["lit-Latn"],
    "ltg": ["ltg-Latn"],
    "lv": ["lav-Latn"],
    "mdf": ["mdf-Cyrl"],
    "mk": ["mkd-Cyrl"],
    "mn": ["mon-Cyrl"],
    "mrj": ["mrj-Cyrl"],
    "mt": ["mlt-Latn"],
    "myv": ["myv-Cyrl"],
    "nan-tw": ["nan-Latn"],
    "nb-NO": ["nob-Latn"],
    "ne-NP": ["nep-Deva"],
    "nhi": ["nhi-Latn"],
    "nl": ["nld-Latn"],
    "nn-NO": ["nno-Latn"],
    "nr": ["nbl-Latn"],
    "nso": ["nso-Latn"],
    "oc": ["oci-Latn"],
    "or": ["ori-Orya"],
    "os": ["oss-Cyrl"],
    "pa-IN": ["pan-Guru"],
    "pl": ["pol-Latn"],
    "ps": ["pus-Arab"],
    "quy": ["quy-Latn"],
    "rm-sursilv": ["roh-Latn"],
    "rm-vallader": ["roh-Latn"],
    "ro": ["ron-Latn"],
    "ru": ["rus-Cyrl"],
    "sah": ["sah-Cyrl"],
    "sat": ["sat-Olck"],
    "sc": ["srd-Latn"],
    "sd": ["snd-Arab"],
    "sk": ["slk-Latn"],
    "skr": ["skr-Arab"],
    "sl": ["slv-Latn"],
    "sq": ["sqi-Latn"],
    "st": ["sot-Latn"],
    "sv-SE": ["swe-Latn"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "ti": ["tir-Ethi"],
    "tig": ["tig-Ethi"],
    "tk": ["tuk-Latn"],
    "tok": ["tok-Latn"],
    "tr": ["tur-Latn"],
    "ts": ["tso-Latn"],
    "tw": ["twi-Latn"],
    "ug": ["uig-Arab"],
    "uk": ["ukr-Cyrl"],
    "vi": ["vie-Latn"],
    "vot": ["vot-Latn"],
    "xh": ["xho-Latn"],
    "yi": ["yid-Hebr"],
    "yo": ["yor-Latn"],
    "yue": ["yue-Hant"],
    "zgh": ["zgh-Tfng"],
    "zh-CN": ["zho-Hans"],
    "zh-HK": ["zho-Hant"],
    "zh-TW": ["zho-Hant"],
    "zu": ["zul-Latn"],
    "zza": ["zza-Latn"],
}


class CommonVoice17A2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice17A2TRetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mteb/common_voice_17_0",
            "revision": "f6564f1e20e5952404a6fed6d65da9b1d393c2d3",
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
            "path": "mteb/common_voice_17_0",
            "revision": "f6564f1e20e5952404a6fed6d65da9b1d393c2d3",
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
