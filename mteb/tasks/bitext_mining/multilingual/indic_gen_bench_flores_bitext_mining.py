from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_LANGUAGES = [
    "ben-Beng",
    "guj_Gujr",
    "hin_Deva",
    "kan_Knda",
    "mal_Mlym",
    "mar_Deva",
    "tam_Taml",
    "tel_Telu",
    "urd_Arab",
    "asm_Beng",
    "bho_Deva",
    "nep_Deva",
    "ory_Orya",
    "pan_Guru",
    "pus_Arab",
    "san-Deva",
    "awa_Deva",
    "bgc_Deva",
    "bod_Tibt",
    "boy_Deva",
    "gbm_Deva",
    "gom_Deva",
    "hne_Deva",
    "raj_Deva",
    "mai_Deva",
    "mni_Mtei",
    "mup_Deva",
    "mwr_Deva",
    "sat_Olck",
]

_ENG_LANGUAGE = ["eng-Latn"]

_CODE_MAPPING = {
    "ben": "bn",
    "guj": "gu",
    "hin": "hi",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "tam": "ta",
    "tel": "te",
    "urd": "ur",
    "asm": "as",
    "bho": "bho",
    "nep": "ne",
    "ory": "or",
    "pan": "pa",
    "pus": "ps",
    "san": "sa",
    "awa": "awa",
    "bgc": "bgc",
    "bod": "bo",
    "boy": "brx",
    "gbm": "gbm",
    "gom": "gom",
    "hne": "hne",
    "raj": "hoj",
    "mai": "mai",
    "mni": "mni",
    "mup": "mup",
    "mwr": "mwr",
    "sat": "sat",
}

_SPLIT = ["validation", "test"]


def get_lang_pairs() -> dict[str, list[str]]:
    # add eng-> xx and xx -> eng lang pairs
    # Normalize language codes
    normalized_languages = [lang.replace("_", "-") for lang in _LANGUAGES]

    # Create dictionary for language pairs
    language_pairs = {}
    for lang in normalized_languages:
        lang_code = lang.split("-")[0]
        lang_to_eng_key = f"{lang_code}-eng"
        eng_to_lang_key = f"eng-{lang_code}"
        language_pairs[lang_to_eng_key] = [lang, _ENG_LANGUAGE[0]]
        language_pairs[eng_to_lang_key] = [_ENG_LANGUAGE[0], lang]

    return language_pairs


_LANGUAGES_MAPPING = get_lang_pairs()


class IndicGenBenchFloresBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="IndicGenBenchFloresBitextMining",
        dataset={
            "path": "mteb/IndicGenBenchFloresBitextMining",
            "revision": "07dcc23c08a2540ba37ebe1e487da9dc497cc15c",
        },
        description="Flores-IN dataset is an extension of Flores dataset released as a part of the IndicGenBench by Google",
        reference="https://github.com/google-research-datasets/indic-gen-bench/",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=_SPLIT,
        eval_langs=_LANGUAGES_MAPPING,
        main_score="f1",
        date=("2023-10-01", "2024-05-01"),
        domains=["Web", "News", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="human-translated and localized",
        bibtex_citation=r"""
@misc{singh2024indicgenbench,
  archiveprefix = {arXiv},
  author = {Harman Singh and Nitish Gupta and Shikhar Bharadwaj and Dinesh Tewari and Partha Talukdar},
  eprint = {2404.16816},
  primaryclass = {cs.CL},
  title = {IndicGenBench: A Multilingual Benchmark to Evaluate Generation Capabilities of LLMs on Indic Languages},
  year = {2024},
}
""",
    )
