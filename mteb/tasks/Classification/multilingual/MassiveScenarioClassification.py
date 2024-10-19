from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "af": ["afr-Latn"],
    "am": ["amh-Ethi"],
    "ar": ["ara-Arab"],
    "az": ["aze-Latn"],
    "bn": ["ben-Beng"],
    "cy": ["cym-Latn"],
    "da": ["dan-Latn"],
    "de": ["deu-Latn"],
    "el": ["ell-Grek"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fa": ["fas-Arab"],
    "fi": ["fin-Latn"],
    "fr": ["fra-Latn"],
    "he": ["heb-Hebr"],
    "hi": ["hin-Deva"],
    "hu": ["hun-Latn"],
    "hy": ["hye-Armn"],
    "id": ["ind-Latn"],
    "is": ["isl-Latn"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "jv": ["jav-Latn"],
    "ka": ["kat-Geor"],
    "km": ["khm-Khmr"],
    "kn": ["kan-Knda"],
    "ko": ["kor-Kore"],
    "lv": ["lav-Latn"],
    "ml": ["mal-Mlym"],
    "mn": ["mon-Cyrl"],
    "ms": ["msa-Latn"],
    "my": ["mya-Mymr"],
    "nb": ["nob-Latn"],
    "nl": ["nld-Latn"],
    "pl": ["pol-Latn"],
    "pt": ["por-Latn"],
    "ro": ["ron-Latn"],
    "ru": ["rus-Cyrl"],
    "sl": ["slv-Latn"],
    "sq": ["sqi-Latn"],
    "sv": ["swe-Latn"],
    "sw": ["swa-Latn"],
    "ta": ["tam-Taml"],
    "te": ["tel-Telu"],
    "th": ["tha-Thai"],
    "tl": ["tgl-Latn"],
    "tr": ["tur-Latn"],
    "ur": ["urd-Arab"],
    "vi": ["vie-Latn"],
    "zh-CN": ["cmo-Hans"],
    "zh-TW": ["cmo-Hant"],
}


class MassiveScenarioClassification(MultilingualTask, AbsTaskClassification):
    fast_loading = True
    metadata = TaskMetadata(
        name="MassiveScenarioClassification",
        dataset={
            "path": "mteb/amazon_massive_scenario",
            "revision": "fad2c6e8459f9e1c45d9315f4953d921437d70f8",
        },
        description="MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages",
        reference="https://arxiv.org/abs/2204.08582",
        category="s2s",
        modalities=["text"],
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2022-01-01", "2022-04-22"),
        domains=["Spoken"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated and localized",  # with the exception of the English data
        bibtex_citation="""@misc{fitzgerald2022massive,
      title={MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages}, 
      author={Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter Leeuwis and Gokhan Tur and Prem Natarajan},
      year={2022},
      eprint={2204.08582},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        prompt="Given a user utterance as query, find the user scenarios",
        descriptive_stats={
            "n_samples": {"validation": 2033, "test": 2974},
            "avg_character_length": {"validation": 34.8, "test": 34.6},
        },
    )
