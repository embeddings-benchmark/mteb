from collections import defaultdict
from typing import Any

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


COMMON_VOICE_CITATION = r"""
@inproceedings{ardila-etal-2020-common,
  archiveprefix = {arXiv},
  eprint = {1912.06670},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  author = {Ardila, Rosana  and
      Branson, Megan  and
      Davis, Kelly  and
      Kohler, Michael  and
      Meyer, Josh  and
      Henretty, Michael  and
      Morais, Reuben  and
      Saunders, Lindsay  and
      Tyers, Francis  and
      Weber, Gregor},
  editor = {Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios},
  booktitle = {Proceedings of the Twelfth Language Resources and Evaluation Conference},
  month = may,
  year = {2020},
  address = {Marseille, France},
  publisher = {European Language Resources Association},
  url = {https://aclanthology.org/2020.lrec-1.520/},
  pages = {4218--4222},
  language = {eng},
  ISBN = {979-10-95546-34-4},
  abstract = {The Common Voice corpus is a massively-multilingual collection of transcribed speech intended for speech technology research and development. Common Voice is designed for Automatic Speech Recognition purposes but can be useful in other domains (e.g. language identification). To achieve scale and sustainability, the Common Voice project employs crowdsourcing for both data collection and data validation. The most recent release includes 29 languages, and as of November 2019 there are a total of 38 languages collecting data. Over 50,000 individuals have participated so far, resulting in 2,500 hours of collected audio. To our knowledge this is the largest audio corpus in the public domain for speech recognition, both in terms of number of hours and number of languages. As an example use case for Common Voice, we present speech recognition experiments using Mozilla{'}s DeepSpeech Speech-to-Text toolkit. By applying transfer learning from a source English model, we find an average Character Error Rate improvement of 5.99 {\ensuremath{\pm}} 5.48 for twelve target languages (German, French, Italian, Turkish, Catalan, Slovenian, Welsh, Irish, Breton, Tatar, Chuvash, and Kabyle). For most of these languages, these are the first ever published results on end-to-end Automatic Speech Recognition.},
}
"""


class CommonVoiceMini17A2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CommonVoiceMini17A2TRetrieval",
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
        main_score="hit_rate_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=COMMON_VOICE_CITATION,
    )

    def load_data(self, **kwargs: Any):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(
        self,
        id_col: str = "path",
        text_col: str = "sentence",
        audio_col: str = "audio",
    ) -> None:
        """Transform Common Voice dataset to MTEB a2t retrieval format.
        Process each language separately to avoid memory accumulation.
        """
        for lang in self.hf_subsets:
            for split in self.metadata.eval_splits:
                # Only load the specific split we need to save memory
                lang_dataset = datasets.load_dataset(
                    self.metadata.dataset["path"],
                    lang,
                    split=split,
                    revision=self.metadata.dataset.get("revision"),
                )

                # Create datasets directly without intermediate lists
                queries_ds = lang_dataset.select_columns(
                    [id_col, audio_col]
                ).rename_column(id_col, "id")

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


class CommonVoiceMini17T2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CommonVoiceMini17T2ARetrieval",
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
        main_score="hit_rate_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=COMMON_VOICE_CITATION,
    )

    def load_data(self, **kwargs: Any):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(
        self,
        id_col: str = "path",
        text_col: str = "sentence",
        audio_col: str = "audio",
    ) -> None:
        """For T2A: query=text, corpus=audio.
        Process each language separately to avoid memory accumulation.
        """
        for lang in self.hf_subsets:
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

                corpus_ds = lang_dataset.select_columns(
                    [id_col, audio_col]
                ).rename_column(id_col, "id")

                relevant_docs_ = {
                    str(row[id_col]): {str(row[id_col]): 1} for row in lang_dataset
                }

                self.corpus[lang][split] = corpus_ds
                self.queries[lang][split] = queries_ds
                self.relevant_docs[lang][split] = relevant_docs_


class CommonVoiceMini21A2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CommonVoiceMini21A2TRetrieval",
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
        main_score="hit_rate_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=COMMON_VOICE_CITATION,
    )

    def load_data(self, **kwargs: Any):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(
        self,
        id_col: str = "path",
        text_col: str = "sentence",
        audio_col: str = "audio",
    ) -> None:
        """Transform Common Voice dataset to MTEB a2t retrieval format.
        Process each language separately to avoid memory accumulation.
        """
        for lang in self.hf_subsets:
            for split in self.metadata.eval_splits:
                # Only load the specific split we need to save memory
                lang_dataset = datasets.load_dataset(
                    self.metadata.dataset["path"],
                    lang,
                    split=split,
                    revision=self.metadata.dataset["revision"],
                )

                # Create datasets directly without intermediate lists
                queries_ds = lang_dataset.select_columns(
                    [id_col, audio_col]
                ).rename_column(id_col, "id")

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


class CommonVoiceMini21T2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CommonVoiceMini21T2ARetrieval",
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
        main_score="hit_rate_at_5",
        date=("2020-01-01", "2024-12-31"),
        domains=["Spoken"],
        task_subtypes=["Speech Retrieval"],
        license="cc0-1.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=COMMON_VOICE_CITATION,
    )

    def load_data(self, **kwargs: Any):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.corpus = defaultdict(DatasetDict)
        self.queries = defaultdict(DatasetDict)
        self.relevant_docs = defaultdict(DatasetDict)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(
        self,
        id_col: str = "path",
        text_col: str = "sentence",
        audio_col: str = "audio",
    ) -> None:
        """For T2A: query=text, corpus=audio.
        Process each language separately to avoid memory accumulation.
        """
        for lang in self.hf_subsets:
            for split in self.metadata.eval_splits:
                lang_dataset = datasets.load_dataset(
                    self.metadata.dataset["path"],
                    lang,
                    split=split,
                    revision=self.metadata.dataset["revision"],
                )

                queries_ds = (
                    lang_dataset.select_columns([id_col, text_col])
                    .rename_column(id_col, "id")
                    .rename_column(text_col, "text")
                )

                corpus_ds = lang_dataset.select_columns(
                    [id_col, audio_col]
                ).rename_column(id_col, "id")

                relevant_docs_ = {
                    str(row[id_col]): {str(row[id_col]): 1} for row in lang_dataset
                }

                self.corpus[lang][split] = corpus_ds
                self.queries[lang][split] = queries_ds
                self.relevant_docs[lang][split] = relevant_docs_
