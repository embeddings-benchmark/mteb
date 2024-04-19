from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, CrosslingualTask

_LANGUAGES = [
    "asm_Beng",
    "ben_Beng",
    "brx_Deva",
    "doi_Deva",
    "eng_Latn",
    "gom_Deva",
    "guj_Gujr",
    "hin_Deva",
    "kan_Knda",
    "kas_Arab",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "mni_Mtei",
    "npi_Deva",
    "ory_Orya",
    "pan_Guru",
    "san_Deva",
    "sat_Olck",
    "snd_Deva",
    "tam_Taml",
    "tel_Telu",
    "urd_Arab",
]
_SPLIT = ["gen"]


def extend_lang_pairs() -> dict[str, list[str]]:
    # add all possible language pairs
    hf_lang_subset2isolang = {}
    for x in _LANGUAGES:
        if "-" not in x:
            for y in _LANGUAGES:
                if x != y:
                    pair = f"{x}-{y}"
                    hf_lang_subset2isolang[pair] = [
                        x.replace("_", "-"),
                        y.replace("_", "-"),
                    ]
    return hf_lang_subset2isolang


_LANGUAGES_MAPPING = extend_lang_pairs()


class IN22GenBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="IN22GenBitextMining",
        dataset={
            "path": "ai4bharat/IN22-Gen",
            "revision": "e92afc34c61104b9b06e4de33cfcaccf6af6a46a",
            "trust_remote_code": True,
        },
        description="IN22-Gen is a general-purpose multi-domain benchmark dataset for machine translation between English and 22 Indic languages.",
        reference="https://huggingface.co/datasets/ai4bharat/IN22-Gen",
        type="BitextMining",
        category="s2s",
        eval_splits=_SPLIT,
        eval_langs=_LANGUAGES_MAPPING,
        main_score="f1",
        date=None,
        form="written",
        domains="Web",
        task_subtypes=None,
        license="CC-BY-4.0",
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=None,
        text_creation="created",
        bibtex_citation=None,
        n_samples={"gen": 1024},
        avg_character_length={},
    )

    def load_data(self, **kwargs: Any) -> None:
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(
                name=lang,
                **self.metadata_dict["dataset"],
            )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        # Convert to standard format
        for lang in self.langs:
            lang1 = lang.split("-")[0]
            lang2 = lang.split("-")[1]
            for split in _SPLIT:
                self.dataset[lang][split] = self.dataset[lang][split].rename_column(
                    "sentence_" + lang1, "sentence1"
                )
                self.dataset[lang][split] = self.dataset[lang][split].rename_column(
                    "sentence_" + lang2, "sentence2"
                )
