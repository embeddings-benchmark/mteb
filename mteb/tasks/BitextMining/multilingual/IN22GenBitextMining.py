from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, MultilingualTask

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
_SPLIT = ["test"]


def extend_lang_pairs() -> dict[str, list[str]]:
    # add all possible language pairs
    hf_lang_subset2isolang = {}
    for x in _LANGUAGES:
        for y in _LANGUAGES:
            if x != y:
                pair = f"{x}-{y}"
                hf_lang_subset2isolang[pair] = [
                    x.replace("_", "-"),
                    y.replace("_", "-"),
                ]
    return hf_lang_subset2isolang


_LANGUAGES_MAPPING = extend_lang_pairs()


class IN22GenBitextMining(AbsTaskBitextMining, MultilingualTask):
    parallel_subsets = True
    metadata = TaskMetadata(
        name="IN22GenBitextMining",
        dataset={
            "path": "mteb/IN22-Gen",
            "revision": "ec381535fe3ddf699297a023bcecaa548096ed68",
            "trust_remote_code": True,
        },
        description="IN22-Gen is a n-way parallel general-purpose multi-domain benchmark dataset for machine translation spanning English and 22 Indic languages.",
        reference="https://huggingface.co/datasets/ai4bharat/IN22-Gen",
        type="BitextMining",
        category="s2s",
        eval_splits=_SPLIT,
        eval_langs=_LANGUAGES_MAPPING,
        main_score="f1",
        date=("2022-10-01", "2023-03-01"),
        form=["written"],
        domains=["Web", "Legal", "Government", "News", "Religious", "Non-fiction"],
        task_subtypes=[],
        license="CC-BY-4.0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@article{gala2023indictrans,
title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
author={Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=vfT4YuzAYA},
note={}
}""",
        n_samples={"test": 1024},
        avg_character_length={"test": 156.7},
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        self.data_loaded = True
