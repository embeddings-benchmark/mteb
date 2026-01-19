from collections import defaultdict

import datasets
from datasets import DatasetDict

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
    "frold": ["fro-Latn"],
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
    "ba": ["bak-Cyrl"],
    "bas": ["bas-Latn"],
    "be": ["bel-Cyrl"],
    "bg": ["bul-Cyrl"],
    "bn": ["ben-Beng"],
    "br": ["bre-Latn"],
    "ca": ["cat-Latn"],
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
    "fa": ["pes-Arab"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "fy-NL": ["fry-Latn"],
    "ga-IE": ["gle-Latn"],
    "gl": ["glg-Latn"],
    "gn": ["grn-Latn"],
    "ha": ["hau-Latn"],
    "he": ["heb-Hebr"],
    "hi": ["hin-Deva"],
    "hsb": ["hsb-Latn"],
    "hu": ["hun-Latn"],
    "hy-AM": ["hye-Armn"],
    "ia": ["ina-Latn"],
    "id": ["ind-Latn"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "ka": ["kat-Geor"],
    "kab": ["kab-Latn"],
    "kk": ["kaz-Cyrl"],
    "kln": ["kln-Latn"],
    "kmr": ["kmr-Latn"],
    "ko": ["kor-Hang"],
    "ky": ["kir-Cyrl"],
    "lg": ["lug-Latn"],
    "lij": ["lij-Latn"],
    "lt": ["lit-Latn"],
    "ltg": ["ltg-Latn"],
    "luo": ["luo-Latn"],
    "lv": ["lav-Latn"],
    "mdf": ["mdf-Cyrl"],
    "mhr": ["mhr-Cyrl"],
    "mk": ["mkd-Cyrl"],
    "ml": ["mal-Mlym"],
    "mn": ["mon-Cyrl"],
    "mr": ["mar-Deva"],
    "mrj": ["mrj-Cyrl"],
    "mt": ["mlt-Latn"],
    "myv": ["myv-Cyrl"],
    "nan-tw": ["nan-Latn"],
    "ne-NP": ["nep-Deva"],
    "nl": ["nld-Latn"],
    "nn-NO": ["nno-Latn"],
    "oc": ["oci-Latn"],
    "or": ["ori-Orya"],
    "os": ["oss-Cyrl"],
    "pa-IN": ["pan-Guru"],
    "pl": ["pol-Latn"],
    "ps": ["pus-Arab"],
    "pt": ["por-Latn"],
    "rm-sursilv": ["roh-Latn"],
    "rm-vallader": ["roh-Latn"],
    "ro": ["ron-Latn"],
    "ru": ["rus-Cyrl"],
    "rw": ["kin-Latn"],
    "sah": ["sah-Cyrl"],
    "sat": ["sat-Olck"],
    "sc": ["srd-Latn"],
    "sk": ["slk-Latn"],
    "skr": ["skr-Arab"],
    "sl": ["slv-Latn"],
    "sq": ["sqi-Latn"],
    "sr": ["srp-Cyrl"],
    "sv-SE": ["swe-Latn"],
    "sw": ["swh-Latn"],
    "ta": ["tam-Taml"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "tig": ["tig-Ethi"],
    "tk": ["tuk-Latn"],
    "tn": ["tsn-Latn"],
    "tok": ["tok-Latn"],
    "tr": ["tur-Latn"],
    "tt": ["tat-Cyrl"],
    "ug": ["uig-Arab"],
    "uk": ["ukr-Cyrl"],
    "ur": ["urd-Arab"],
    "uz": ["uzn-Latn"],
    "vi": ["vie-Latn"],
    "yi": ["yid-Hebr"],
    "yo": ["yor-Latn"],
    "yue": ["yue-Hant"],
    "zgh": ["zgh-Tfng"],
    "zh-CN": ["zho-Hans"],
    "zh-HK": ["zho-Hant"],
    "zh-TW": ["zho-Hant"],
    "zza": ["zza-Latn"],
}


class CommonVoice17A2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice17A2TRetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mteb/common_voice_17_0_mini",
            "revision": "62c6f7bb73d8cdb684868c14620ce241448e471b",
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
        """Transform Common Voice dataset to MTEB a2t retrieval format.
        Process each language separately to avoid memory accumulation.
        """
        for lang in self.metadata.eval_langs:
            for split in self.metadata.eval_splits:
                # Only load the specific split we need to save memory
                lang_dataset = datasets.load_dataset(
                    self.metadata.dataset["path"],
                    lang,
                    split=split,
                    revision=self.metadata.dataset.get("revision"),
                )

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

                self.corpus[lang][split] = corpus_ds
                self.queries[lang][split] = queries_ds
                self.relevant_docs[lang][split] = relevant_docs_


class CommonVoice17T2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice17T2ARetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mteb/common_voice_17_0_mini",
            "revision": "62c6f7bb73d8cdb684868c14620ce241448e471b",
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
        """For T2A: query=text, corpus=audio.
        Process each language separately to avoid memory accumulation.
        """
        for lang in self.metadata.eval_langs:
            for split in self.metadata.eval_splits:
                lang_dataset = datasets.load_dataset(
                    self.metadata.dataset["path"],
                    lang,
                    split=split,
                    revision=self.metadata.dataset.get("revision"),
                )

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

                self.corpus[lang][split] = corpus_ds
                self.queries[lang][split] = queries_ds
                self.relevant_docs[lang][split] = relevant_docs_


class CommonVoice21A2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice21A2TRetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mteb/common_voice_21_0_mini",
            "revision": "8aef059b329d70e590bb81454a8a85ecdae54b45",
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
        """Transform Common Voice dataset to MTEB a2t retrieval format.
        Process each language separately to avoid memory accumulation.
        """
        for lang in self.metadata.eval_langs:
            for split in self.metadata.eval_splits:
                # Only load the specific split we need to save memory
                lang_dataset = datasets.load_dataset(
                    self.metadata.dataset["path"],
                    lang,
                    split=split,
                    revision=self.metadata.dataset.get("revision"),
                )

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

                self.corpus[lang][split] = corpus_ds
                self.queries[lang][split] = queries_ds
                self.relevant_docs[lang][split] = relevant_docs_


class CommonVoice21T2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CommonVoice21T2ARetrieval",
        description="Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
        reference="https://commonvoice.mozilla.org/",
        dataset={
            "path": "mteb/common_voice_21_0_mini",
            "revision": "8aef059b329d70e590bb81454a8a85ecdae54b45",
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
        """For T2A: query=text, corpus=audio.
        Process each language separately to avoid memory accumulation.
        """
        for lang in self.metadata.eval_langs:
            for split in self.metadata.eval_splits:
                lang_dataset = datasets.load_dataset(
                    self.metadata.dataset["path"],
                    lang,
                    split=split,
                    revision=self.metadata.dataset.get("revision"),
                )

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

                self.corpus[lang][split] = corpus_ds
                self.queries[lang][split] = queries_ds
                self.relevant_docs[lang][split] = relevant_docs_
