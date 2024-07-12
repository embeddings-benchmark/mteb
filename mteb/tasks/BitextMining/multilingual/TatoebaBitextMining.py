from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, MultilingualTask

_LANGUAGES = {
    "sqi-eng": ["sqi-Latn", "eng-Latn"],
    "fry-eng": ["fry-Latn", "eng-Latn"],
    "kur-eng": ["kur-Latn", "eng-Latn"],
    "tur-eng": ["tur-Latn", "eng-Latn"],
    "deu-eng": ["deu-Latn", "eng-Latn"],
    "nld-eng": ["nld-Latn", "eng-Latn"],
    "ron-eng": ["ron-Latn", "eng-Latn"],
    "ang-eng": ["ang-Latn", "eng-Latn"],
    "ido-eng": ["ido-Latn", "eng-Latn"],
    "jav-eng": ["jav-Latn", "eng-Latn"],
    "isl-eng": ["isl-Latn", "eng-Latn"],
    "slv-eng": ["slv-Latn", "eng-Latn"],
    "cym-eng": ["cym-Latn", "eng-Latn"],
    "kaz-eng": ["kaz-Cyrl", "eng-Latn"],
    "est-eng": ["est-Latn", "eng-Latn"],
    "heb-eng": ["heb-Hebr", "eng-Latn"],
    "gla-eng": ["gla-Latn", "eng-Latn"],
    "mar-eng": ["mar-Deva", "eng-Latn"],
    "lat-eng": ["lat-Latn", "eng-Latn"],
    "bel-eng": ["bel-Cyrl", "eng-Latn"],
    "pms-eng": ["pms-Latn", "eng-Latn"],
    "gle-eng": ["gle-Latn", "eng-Latn"],
    "pes-eng": ["pes-Arab", "eng-Latn"],
    "nob-eng": ["nob-Latn", "eng-Latn"],
    "bul-eng": ["bul-Cyrl", "eng-Latn"],
    "cbk-eng": ["cbk-Latn", "eng-Latn"],
    "hun-eng": ["hun-Latn", "eng-Latn"],
    "uig-eng": ["uig-Arab", "eng-Latn"],
    "rus-eng": ["rus-Cyrl", "eng-Latn"],
    "spa-eng": ["spa-Latn", "eng-Latn"],
    "hye-eng": ["hye-Armn", "eng-Latn"],
    "tel-eng": ["tel-Telu", "eng-Latn"],
    "afr-eng": ["afr-Latn", "eng-Latn"],
    "mon-eng": ["mon-Cyrl", "eng-Latn"],
    "arz-eng": ["arz-Arab", "eng-Latn"],
    "hrv-eng": ["hrv-Latn", "eng-Latn"],
    "nov-eng": ["nov-Latn", "eng-Latn"],
    "gsw-eng": ["gsw-Latn", "eng-Latn"],
    "nds-eng": ["nds-Latn", "eng-Latn"],
    "ukr-eng": ["ukr-Cyrl", "eng-Latn"],
    "uzb-eng": ["uzb-Latn", "eng-Latn"],
    "lit-eng": ["lit-Latn", "eng-Latn"],
    "ina-eng": ["ina-Latn", "eng-Latn"],
    "lfn-eng": ["lfn-Latn", "eng-Latn"],
    "zsm-eng": ["zsm-Latn", "eng-Latn"],
    "ita-eng": ["ita-Latn", "eng-Latn"],
    "cmn-eng": ["cmn-Hans", "eng-Latn"],
    "lvs-eng": ["lvs-Latn", "eng-Latn"],
    "glg-eng": ["glg-Latn", "eng-Latn"],
    "ceb-eng": ["ceb-Latn", "eng-Latn"],
    "bre-eng": ["bre-Latn", "eng-Latn"],
    "ben-eng": ["ben-Beng", "eng-Latn"],
    "swg-eng": ["swg-Latn", "eng-Latn"],
    "arq-eng": ["arq-Arab", "eng-Latn"],
    "kab-eng": ["kab-Latn", "eng-Latn"],
    "fra-eng": ["fra-Latn", "eng-Latn"],
    "por-eng": ["por-Latn", "eng-Latn"],
    "tat-eng": ["tat-Cyrl", "eng-Latn"],
    "oci-eng": ["oci-Latn", "eng-Latn"],
    "pol-eng": ["pol-Latn", "eng-Latn"],
    "war-eng": ["war-Latn", "eng-Latn"],
    "aze-eng": ["aze-Latn", "eng-Latn"],
    "vie-eng": ["vie-Latn", "eng-Latn"],
    "nno-eng": ["nno-Latn", "eng-Latn"],
    "cha-eng": ["cha-Latn", "eng-Latn"],
    "mhr-eng": ["mhr-Cyrl", "eng-Latn"],
    "dan-eng": ["dan-Latn", "eng-Latn"],
    "ell-eng": ["ell-Grek", "eng-Latn"],
    "amh-eng": ["amh-Ethi", "eng-Latn"],
    "pam-eng": ["pam-Latn", "eng-Latn"],
    "hsb-eng": ["hsb-Latn", "eng-Latn"],
    "srp-eng": ["srp-Cyrl", "eng-Latn"],
    "epo-eng": ["epo-Latn", "eng-Latn"],
    "kzj-eng": ["kzj-Latn", "eng-Latn"],
    "awa-eng": ["awa-Deva", "eng-Latn"],
    "fao-eng": ["fao-Latn", "eng-Latn"],
    "mal-eng": ["mal-Mlym", "eng-Latn"],
    "ile-eng": ["ile-Latn", "eng-Latn"],
    "bos-eng": ["bos-Latn", "eng-Latn"],
    "cor-eng": ["cor-Latn", "eng-Latn"],
    "cat-eng": ["cat-Latn", "eng-Latn"],
    "eus-eng": ["eus-Latn", "eng-Latn"],
    "yue-eng": ["yue-Hant", "eng-Latn"],
    "swe-eng": ["swe-Latn", "eng-Latn"],
    "dtp-eng": ["dtp-Latn", "eng-Latn"],
    "kat-eng": ["kat-Geor", "eng-Latn"],
    "jpn-eng": ["jpn-Jpan", "eng-Latn"],
    "csb-eng": ["csb-Latn", "eng-Latn"],
    "xho-eng": ["xho-Latn", "eng-Latn"],
    "orv-eng": ["orv-Cyrl", "eng-Latn"],
    "ind-eng": ["ind-Latn", "eng-Latn"],
    "tuk-eng": ["tuk-Latn", "eng-Latn"],
    "max-eng": ["max-Deva", "eng-Latn"],
    "swh-eng": ["swh-Latn", "eng-Latn"],
    "hin-eng": ["hin-Deva", "eng-Latn"],
    "dsb-eng": ["dsb-Latn", "eng-Latn"],
    "ber-eng": ["ber-Tfng", "eng-Latn"],
    "tam-eng": ["tam-Taml", "eng-Latn"],
    "slk-eng": ["slk-Latn", "eng-Latn"],
    "tgl-eng": ["tgl-Latn", "eng-Latn"],
    "ast-eng": ["ast-Latn", "eng-Latn"],
    "mkd-eng": ["mkd-Cyrl", "eng-Latn"],
    "khm-eng": ["khm-Khmr", "eng-Latn"],
    "ces-eng": ["ces-Latn", "eng-Latn"],
    "tzl-eng": ["tzl-Latn", "eng-Latn"],
    "urd-eng": ["urd-Arab", "eng-Latn"],
    "ara-eng": ["ara-Arab", "eng-Latn"],
    "kor-eng": ["kor-Hang", "eng-Latn"],
    "yid-eng": ["yid-Hebr", "eng-Latn"],
    "fin-eng": ["fin-Latn", "eng-Latn"],
    "tha-eng": ["tha-Thai", "eng-Latn"],
    "wuu-eng": ["wuu-Hans", "eng-Latn"],
}


class TatoebaBitextMining(AbsTaskBitextMining, MultilingualTask):
    fast_loading = True
    metadata = TaskMetadata(
        name="Tatoeba",
        dataset={
            "path": "mteb/tatoeba-bitext-mining",
            "revision": "69e8f12da6e31d59addadda9a9c8a2e601a0e282",
        },
        description="1,000 English-aligned sentence pairs for each language based on the Tatoeba corpus",
        reference="https://github.com/facebookresearch/LASER/tree/main/data/tatoeba/v1",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2006-01-01", "2021-12-31"),  # Estimated range
        domains=[
            "Written"
        ],  # Tatoeba corpus includes a wide range of topics and domains
        task_subtypes=[],
        license="CC BY 2.0",
        annotations_creators="human-annotated",
        dialect=[],  # No specific dialect mentioned
        sample_creation="found",
        bibtex_citation="""
        @misc{tatoeba,
        author = {Tatoeba community},
        title = {Tatoeba: Collection of sentences and translations},
        year = {2021},
        }
        """,
        descriptive_stats={
            "n_samples": {"test": 2000},
            "avg_character_length": {"test": 39.4},
        },
    )
