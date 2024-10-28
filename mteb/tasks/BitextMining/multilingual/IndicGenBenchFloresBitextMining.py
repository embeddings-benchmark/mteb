from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

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


class IndicGenBenchFloresBitextMining(AbsTaskBitextMining, MultilingualTask):
    metadata = TaskMetadata(
        name="IndicGenBenchFloresBitextMining",
        dataset={
            "path": "google/IndicGenBench_flores_in",
            "revision": "f8650438298df086750ff4973661bb58a201a5ee",
            "trust_remote_code": True,
        },
        description="Flores-IN dataset is an extension of Flores dataset released as a part of the IndicGenBench by Google",
        reference="https://github.com/google-research-datasets/indic-gen-bench/",
        type="BitextMining",
        category="s2s",
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
        bibtex_citation="""@misc{singh2024indicgenbench,
      title={IndicGenBench: A Multilingual Benchmark to Evaluate Generation Capabilities of LLMs on Indic Languages}, 
      author={Harman Singh and Nitish Gupta and Shikhar Bharadwaj and Dinesh Tewari and Partha Talukdar},
      year={2024},
      eprint={2404.16816},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        descriptive_stats={
            "n_samples": {"validation": 997, "test": 1012},
            "avg_character_length": {"validation": 126.25, "test": 130.84},
        },
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return

        self.dataset = {}
        for lang in self.hf_subsets:
            langs = lang.split("-")
            source_lang = langs[0]
            target_lang = langs[1]
            if source_lang == "eng":
                coded_target_language = _CODE_MAPPING[target_lang]
                language = f"en_{coded_target_language}"

            else:
                coded_source_language = _CODE_MAPPING[source_lang]
                language = f"{coded_source_language}_en"

            self.dataset[lang] = datasets.load_dataset(
                **self.metadata_dict["dataset"],
                field="examples",
                data_files={
                    "validation": f"flores_{language}_dev.json",
                    "test": f"flores_{language}_test.json",
                },
            )

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        for lang in self.hf_subsets:
            for split in _SPLIT:
                self.dataset[lang][split] = self.dataset[lang][split].rename_columns(
                    {"source": "sentence1", "target": "sentence2"}
                )
