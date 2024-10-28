from __future__ import annotations

from typing import Any

import datasets

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

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


def get_hash(text):
    """Get hash of text field."""
    return {"hash": hash(text)}


def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False


class IN22ConvBitextMining(AbsTaskBitextMining, MultilingualTask):
    parallel_subsets = True
    metadata = TaskMetadata(
        name="IN22ConvBitextMining",
        dataset={
            "path": "mteb/IN22-Conv",
            "revision": "16f46f059d56eac7c65c3c9581a45e40199eb140",
            "trust_remote_code": True,
        },
        description="IN22-Conv is a n-way parallel conversation domain benchmark dataset for machine translation spanning English and 22 Indic languages.",
        reference="https://huggingface.co/datasets/ai4bharat/IN22-Conv",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=_SPLIT,
        eval_langs=_LANGUAGES_MAPPING,
        main_score="f1",
        date=("2022-10-01", "2023-03-01"),
        domains=["Social", "Spoken", "Fiction", "Spoken"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@article{gala2023indictrans,
title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
author={Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=vfT4YuzAYA},
note={}
}""",
        descriptive_stats={
            "test": {
                "average_sentence1_length": 54.32948595562498,
                "average_sentence2_length": 54.32948595562498,
                "num_samples": 760518,
                "hf_subset_descriptive_stats": {
                    "asm_Beng-ben_Beng": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "asm_Beng-brx_Deva": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "asm_Beng-doi_Deva": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "asm_Beng-eng_Latn": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "asm_Beng-gom_Deva": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "asm_Beng-guj_Gujr": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "asm_Beng-hin_Deva": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "asm_Beng-kan_Knda": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "asm_Beng-kas_Arab": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "asm_Beng-mai_Deva": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "asm_Beng-mal_Mlym": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "asm_Beng-mar_Deva": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "asm_Beng-mni_Mtei": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "asm_Beng-npi_Deva": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "asm_Beng-ory_Orya": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "asm_Beng-pan_Guru": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "asm_Beng-san_Deva": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "asm_Beng-sat_Olck": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "asm_Beng-snd_Deva": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "asm_Beng-tam_Taml": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "asm_Beng-tel_Telu": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "asm_Beng-urd_Arab": {
                        "average_sentence1_length": 53.753825681969396,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "ben_Beng-asm_Beng": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "ben_Beng-brx_Deva": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "ben_Beng-doi_Deva": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "ben_Beng-eng_Latn": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "ben_Beng-gom_Deva": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "ben_Beng-guj_Gujr": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "ben_Beng-hin_Deva": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "ben_Beng-kan_Knda": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "ben_Beng-kas_Arab": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "ben_Beng-mai_Deva": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "ben_Beng-mal_Mlym": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "ben_Beng-mar_Deva": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "ben_Beng-mni_Mtei": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "ben_Beng-npi_Deva": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "ben_Beng-ory_Orya": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "ben_Beng-pan_Guru": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "ben_Beng-san_Deva": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "ben_Beng-sat_Olck": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "ben_Beng-snd_Deva": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "ben_Beng-tam_Taml": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "ben_Beng-tel_Telu": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "ben_Beng-urd_Arab": {
                        "average_sentence1_length": 50.03060545575516,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "brx_Deva-asm_Beng": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "brx_Deva-ben_Beng": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "brx_Deva-doi_Deva": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "brx_Deva-eng_Latn": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "brx_Deva-gom_Deva": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "brx_Deva-guj_Gujr": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "brx_Deva-hin_Deva": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "brx_Deva-kan_Knda": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "brx_Deva-kas_Arab": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "brx_Deva-mai_Deva": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "brx_Deva-mal_Mlym": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "brx_Deva-mar_Deva": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "brx_Deva-mni_Mtei": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "brx_Deva-npi_Deva": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "brx_Deva-ory_Orya": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "brx_Deva-pan_Guru": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "brx_Deva-san_Deva": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "brx_Deva-sat_Olck": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "brx_Deva-snd_Deva": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "brx_Deva-tam_Taml": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "brx_Deva-tel_Telu": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "brx_Deva-urd_Arab": {
                        "average_sentence1_length": 54.05988023952096,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "doi_Deva-asm_Beng": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "doi_Deva-ben_Beng": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "doi_Deva-brx_Deva": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "doi_Deva-eng_Latn": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "doi_Deva-gom_Deva": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "doi_Deva-guj_Gujr": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "doi_Deva-hin_Deva": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "doi_Deva-kan_Knda": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "doi_Deva-kas_Arab": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "doi_Deva-mai_Deva": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "doi_Deva-mal_Mlym": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "doi_Deva-mar_Deva": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "doi_Deva-mni_Mtei": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "doi_Deva-npi_Deva": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "doi_Deva-ory_Orya": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "doi_Deva-pan_Guru": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "doi_Deva-san_Deva": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "doi_Deva-sat_Olck": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "doi_Deva-snd_Deva": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "doi_Deva-tam_Taml": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "doi_Deva-tel_Telu": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "doi_Deva-urd_Arab": {
                        "average_sentence1_length": 57.37857618097139,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "eng_Latn-asm_Beng": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "eng_Latn-ben_Beng": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "eng_Latn-brx_Deva": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "eng_Latn-doi_Deva": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "eng_Latn-gom_Deva": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "eng_Latn-guj_Gujr": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "eng_Latn-hin_Deva": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "eng_Latn-kan_Knda": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "eng_Latn-kas_Arab": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "eng_Latn-mai_Deva": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "eng_Latn-mal_Mlym": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "eng_Latn-mar_Deva": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "eng_Latn-mni_Mtei": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "eng_Latn-npi_Deva": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "eng_Latn-ory_Orya": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "eng_Latn-pan_Guru": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "eng_Latn-san_Deva": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "eng_Latn-sat_Olck": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "eng_Latn-snd_Deva": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "eng_Latn-tam_Taml": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "eng_Latn-tel_Telu": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "eng_Latn-urd_Arab": {
                        "average_sentence1_length": 53.17631403858949,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "gom_Deva-asm_Beng": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "gom_Deva-ben_Beng": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "gom_Deva-brx_Deva": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "gom_Deva-doi_Deva": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "gom_Deva-eng_Latn": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "gom_Deva-guj_Gujr": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "gom_Deva-hin_Deva": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "gom_Deva-kan_Knda": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "gom_Deva-kas_Arab": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "gom_Deva-mai_Deva": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "gom_Deva-mal_Mlym": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "gom_Deva-mar_Deva": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "gom_Deva-mni_Mtei": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "gom_Deva-npi_Deva": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "gom_Deva-ory_Orya": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "gom_Deva-pan_Guru": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "gom_Deva-san_Deva": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "gom_Deva-sat_Olck": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "gom_Deva-snd_Deva": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "gom_Deva-tam_Taml": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "gom_Deva-tel_Telu": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "gom_Deva-urd_Arab": {
                        "average_sentence1_length": 50.22621423819029,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-asm_Beng": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-ben_Beng": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-brx_Deva": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-doi_Deva": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-eng_Latn": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-gom_Deva": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-hin_Deva": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-kan_Knda": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-kas_Arab": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-mai_Deva": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-mal_Mlym": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-mar_Deva": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-mni_Mtei": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-npi_Deva": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-ory_Orya": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-pan_Guru": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-san_Deva": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-sat_Olck": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-snd_Deva": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-tam_Taml": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-tel_Telu": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "guj_Gujr-urd_Arab": {
                        "average_sentence1_length": 51.54823685961411,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "hin_Deva-asm_Beng": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "hin_Deva-ben_Beng": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "hin_Deva-brx_Deva": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "hin_Deva-doi_Deva": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "hin_Deva-eng_Latn": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "hin_Deva-gom_Deva": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "hin_Deva-guj_Gujr": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "hin_Deva-kan_Knda": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "hin_Deva-kas_Arab": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "hin_Deva-mai_Deva": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "hin_Deva-mal_Mlym": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "hin_Deva-mar_Deva": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "hin_Deva-mni_Mtei": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "hin_Deva-npi_Deva": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "hin_Deva-ory_Orya": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "hin_Deva-pan_Guru": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "hin_Deva-san_Deva": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "hin_Deva-sat_Olck": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "hin_Deva-snd_Deva": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "hin_Deva-tam_Taml": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "hin_Deva-tel_Telu": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "hin_Deva-urd_Arab": {
                        "average_sentence1_length": 52.67598137059215,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "kan_Knda-asm_Beng": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "kan_Knda-ben_Beng": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "kan_Knda-brx_Deva": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "kan_Knda-doi_Deva": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "kan_Knda-eng_Latn": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "kan_Knda-gom_Deva": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "kan_Knda-guj_Gujr": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "kan_Knda-hin_Deva": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "kan_Knda-kas_Arab": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "kan_Knda-mai_Deva": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "kan_Knda-mal_Mlym": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "kan_Knda-mar_Deva": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "kan_Knda-mni_Mtei": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "kan_Knda-npi_Deva": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "kan_Knda-ory_Orya": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "kan_Knda-pan_Guru": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "kan_Knda-san_Deva": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "kan_Knda-sat_Olck": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "kan_Knda-snd_Deva": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "kan_Knda-tam_Taml": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "kan_Knda-tel_Telu": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "kan_Knda-urd_Arab": {
                        "average_sentence1_length": 56.14437791084497,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "kas_Arab-asm_Beng": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "kas_Arab-ben_Beng": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "kas_Arab-brx_Deva": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "kas_Arab-doi_Deva": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "kas_Arab-eng_Latn": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "kas_Arab-gom_Deva": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "kas_Arab-guj_Gujr": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "kas_Arab-hin_Deva": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "kas_Arab-kan_Knda": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "kas_Arab-mai_Deva": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "kas_Arab-mal_Mlym": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "kas_Arab-mar_Deva": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "kas_Arab-mni_Mtei": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "kas_Arab-npi_Deva": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "kas_Arab-ory_Orya": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "kas_Arab-pan_Guru": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "kas_Arab-san_Deva": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "kas_Arab-sat_Olck": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "kas_Arab-snd_Deva": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "kas_Arab-tam_Taml": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "kas_Arab-tel_Telu": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "kas_Arab-urd_Arab": {
                        "average_sentence1_length": 55.81437125748503,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "mai_Deva-asm_Beng": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "mai_Deva-ben_Beng": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "mai_Deva-brx_Deva": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "mai_Deva-doi_Deva": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "mai_Deva-eng_Latn": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "mai_Deva-gom_Deva": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "mai_Deva-guj_Gujr": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "mai_Deva-hin_Deva": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "mai_Deva-kan_Knda": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "mai_Deva-kas_Arab": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "mai_Deva-mal_Mlym": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "mai_Deva-mar_Deva": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "mai_Deva-mni_Mtei": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "mai_Deva-npi_Deva": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "mai_Deva-ory_Orya": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "mai_Deva-pan_Guru": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "mai_Deva-san_Deva": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "mai_Deva-sat_Olck": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "mai_Deva-snd_Deva": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "mai_Deva-tam_Taml": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "mai_Deva-tel_Telu": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "mai_Deva-urd_Arab": {
                        "average_sentence1_length": 54.3020625415835,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-asm_Beng": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-ben_Beng": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-brx_Deva": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-doi_Deva": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-eng_Latn": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-gom_Deva": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-guj_Gujr": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-hin_Deva": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-kan_Knda": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-kas_Arab": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-mai_Deva": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-mar_Deva": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-mni_Mtei": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-npi_Deva": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-ory_Orya": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-pan_Guru": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-san_Deva": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-sat_Olck": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-snd_Deva": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-tam_Taml": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-tel_Telu": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "mal_Mlym-urd_Arab": {
                        "average_sentence1_length": 61.24151696606786,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "mar_Deva-asm_Beng": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "mar_Deva-ben_Beng": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "mar_Deva-brx_Deva": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "mar_Deva-doi_Deva": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "mar_Deva-eng_Latn": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "mar_Deva-gom_Deva": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "mar_Deva-guj_Gujr": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "mar_Deva-hin_Deva": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "mar_Deva-kan_Knda": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "mar_Deva-kas_Arab": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "mar_Deva-mai_Deva": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "mar_Deva-mal_Mlym": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "mar_Deva-mni_Mtei": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "mar_Deva-npi_Deva": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "mar_Deva-ory_Orya": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "mar_Deva-pan_Guru": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "mar_Deva-san_Deva": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "mar_Deva-sat_Olck": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "mar_Deva-snd_Deva": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "mar_Deva-tam_Taml": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "mar_Deva-tel_Telu": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "mar_Deva-urd_Arab": {
                        "average_sentence1_length": 54.52761144377911,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-asm_Beng": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-ben_Beng": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-brx_Deva": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-doi_Deva": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-eng_Latn": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-gom_Deva": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-guj_Gujr": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-hin_Deva": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-kan_Knda": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-kas_Arab": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-mai_Deva": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-mal_Mlym": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-mar_Deva": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-npi_Deva": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-ory_Orya": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-pan_Guru": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-san_Deva": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-sat_Olck": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-snd_Deva": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-tam_Taml": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-tel_Telu": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "mni_Mtei-urd_Arab": {
                        "average_sentence1_length": 50.91417165668663,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "npi_Deva-asm_Beng": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "npi_Deva-ben_Beng": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "npi_Deva-brx_Deva": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "npi_Deva-doi_Deva": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "npi_Deva-eng_Latn": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "npi_Deva-gom_Deva": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "npi_Deva-guj_Gujr": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "npi_Deva-hin_Deva": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "npi_Deva-kan_Knda": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "npi_Deva-kas_Arab": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "npi_Deva-mai_Deva": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "npi_Deva-mal_Mlym": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "npi_Deva-mar_Deva": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "npi_Deva-mni_Mtei": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "npi_Deva-ory_Orya": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "npi_Deva-pan_Guru": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "npi_Deva-san_Deva": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "npi_Deva-sat_Olck": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "npi_Deva-snd_Deva": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "npi_Deva-tam_Taml": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "npi_Deva-tel_Telu": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "npi_Deva-urd_Arab": {
                        "average_sentence1_length": 53.30272787757818,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "ory_Orya-asm_Beng": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "ory_Orya-ben_Beng": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "ory_Orya-brx_Deva": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "ory_Orya-doi_Deva": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "ory_Orya-eng_Latn": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "ory_Orya-gom_Deva": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "ory_Orya-guj_Gujr": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "ory_Orya-hin_Deva": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "ory_Orya-kan_Knda": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "ory_Orya-kas_Arab": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "ory_Orya-mai_Deva": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "ory_Orya-mal_Mlym": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "ory_Orya-mar_Deva": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "ory_Orya-mni_Mtei": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "ory_Orya-npi_Deva": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "ory_Orya-pan_Guru": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "ory_Orya-san_Deva": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "ory_Orya-sat_Olck": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "ory_Orya-snd_Deva": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "ory_Orya-tam_Taml": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "ory_Orya-tel_Telu": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "ory_Orya-urd_Arab": {
                        "average_sentence1_length": 55.509647371922824,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "pan_Guru-asm_Beng": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "pan_Guru-ben_Beng": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "pan_Guru-brx_Deva": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "pan_Guru-doi_Deva": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "pan_Guru-eng_Latn": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "pan_Guru-gom_Deva": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "pan_Guru-guj_Gujr": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "pan_Guru-hin_Deva": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "pan_Guru-kan_Knda": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "pan_Guru-kas_Arab": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "pan_Guru-mai_Deva": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "pan_Guru-mal_Mlym": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "pan_Guru-mar_Deva": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "pan_Guru-mni_Mtei": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "pan_Guru-npi_Deva": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "pan_Guru-ory_Orya": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "pan_Guru-san_Deva": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "pan_Guru-sat_Olck": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "pan_Guru-snd_Deva": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "pan_Guru-tam_Taml": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "pan_Guru-tel_Telu": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "pan_Guru-urd_Arab": {
                        "average_sentence1_length": 52.83366600133067,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "san_Deva-asm_Beng": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "san_Deva-ben_Beng": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "san_Deva-brx_Deva": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "san_Deva-doi_Deva": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "san_Deva-eng_Latn": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "san_Deva-gom_Deva": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "san_Deva-guj_Gujr": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "san_Deva-hin_Deva": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "san_Deva-kan_Knda": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "san_Deva-kas_Arab": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "san_Deva-mai_Deva": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "san_Deva-mal_Mlym": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "san_Deva-mar_Deva": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "san_Deva-mni_Mtei": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "san_Deva-npi_Deva": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "san_Deva-ory_Orya": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "san_Deva-pan_Guru": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "san_Deva-sat_Olck": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "san_Deva-snd_Deva": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "san_Deva-tam_Taml": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "san_Deva-tel_Telu": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "san_Deva-urd_Arab": {
                        "average_sentence1_length": 51.4311377245509,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "sat_Olck-asm_Beng": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "sat_Olck-ben_Beng": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "sat_Olck-brx_Deva": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "sat_Olck-doi_Deva": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "sat_Olck-eng_Latn": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "sat_Olck-gom_Deva": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "sat_Olck-guj_Gujr": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "sat_Olck-hin_Deva": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "sat_Olck-kan_Knda": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "sat_Olck-kas_Arab": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "sat_Olck-mai_Deva": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "sat_Olck-mal_Mlym": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "sat_Olck-mar_Deva": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "sat_Olck-mni_Mtei": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "sat_Olck-npi_Deva": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "sat_Olck-ory_Orya": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "sat_Olck-pan_Guru": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "sat_Olck-san_Deva": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "sat_Olck-snd_Deva": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "sat_Olck-tam_Taml": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "sat_Olck-tel_Telu": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "sat_Olck-urd_Arab": {
                        "average_sentence1_length": 58.94011976047904,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "snd_Deva-asm_Beng": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "snd_Deva-ben_Beng": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "snd_Deva-brx_Deva": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "snd_Deva-doi_Deva": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "snd_Deva-eng_Latn": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "snd_Deva-gom_Deva": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "snd_Deva-guj_Gujr": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "snd_Deva-hin_Deva": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "snd_Deva-kan_Knda": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "snd_Deva-kas_Arab": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "snd_Deva-mai_Deva": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "snd_Deva-mal_Mlym": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "snd_Deva-mar_Deva": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "snd_Deva-mni_Mtei": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "snd_Deva-npi_Deva": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "snd_Deva-ory_Orya": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "snd_Deva-pan_Guru": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "snd_Deva-san_Deva": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "snd_Deva-sat_Olck": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "snd_Deva-tam_Taml": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "snd_Deva-tel_Telu": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "snd_Deva-urd_Arab": {
                        "average_sentence1_length": 54.445109780439125,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "tam_Taml-asm_Beng": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "tam_Taml-ben_Beng": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "tam_Taml-brx_Deva": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "tam_Taml-doi_Deva": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "tam_Taml-eng_Latn": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "tam_Taml-gom_Deva": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "tam_Taml-guj_Gujr": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "tam_Taml-hin_Deva": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "tam_Taml-kan_Knda": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "tam_Taml-kas_Arab": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "tam_Taml-mai_Deva": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "tam_Taml-mal_Mlym": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "tam_Taml-mar_Deva": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "tam_Taml-mni_Mtei": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "tam_Taml-npi_Deva": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "tam_Taml-ory_Orya": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "tam_Taml-pan_Guru": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "tam_Taml-san_Deva": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "tam_Taml-sat_Olck": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "tam_Taml-snd_Deva": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "tam_Taml-tel_Telu": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                    "tam_Taml-urd_Arab": {
                        "average_sentence1_length": 62.590818363273456,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "tel_Telu-asm_Beng": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "tel_Telu-ben_Beng": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "tel_Telu-brx_Deva": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "tel_Telu-doi_Deva": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "tel_Telu-eng_Latn": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "tel_Telu-gom_Deva": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "tel_Telu-guj_Gujr": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "tel_Telu-hin_Deva": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "tel_Telu-kan_Knda": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "tel_Telu-kas_Arab": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "tel_Telu-mai_Deva": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "tel_Telu-mal_Mlym": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "tel_Telu-mar_Deva": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "tel_Telu-mni_Mtei": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "tel_Telu-npi_Deva": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "tel_Telu-ory_Orya": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "tel_Telu-pan_Guru": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "tel_Telu-san_Deva": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "tel_Telu-sat_Olck": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "tel_Telu-snd_Deva": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "tel_Telu-tam_Taml": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "tel_Telu-urd_Arab": {
                        "average_sentence1_length": 51.16300731869594,
                        "average_sentence2_length": 53.568196939454424,
                        "num_samples": 1503,
                    },
                    "urd_Arab-asm_Beng": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 53.753825681969396,
                        "num_samples": 1503,
                    },
                    "urd_Arab-ben_Beng": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 50.03060545575516,
                        "num_samples": 1503,
                    },
                    "urd_Arab-brx_Deva": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 54.05988023952096,
                        "num_samples": 1503,
                    },
                    "urd_Arab-doi_Deva": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 57.37857618097139,
                        "num_samples": 1503,
                    },
                    "urd_Arab-eng_Latn": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 53.17631403858949,
                        "num_samples": 1503,
                    },
                    "urd_Arab-gom_Deva": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 50.22621423819029,
                        "num_samples": 1503,
                    },
                    "urd_Arab-guj_Gujr": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 51.54823685961411,
                        "num_samples": 1503,
                    },
                    "urd_Arab-hin_Deva": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 52.67598137059215,
                        "num_samples": 1503,
                    },
                    "urd_Arab-kan_Knda": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 56.14437791084497,
                        "num_samples": 1503,
                    },
                    "urd_Arab-kas_Arab": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 55.81437125748503,
                        "num_samples": 1503,
                    },
                    "urd_Arab-mai_Deva": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 54.3020625415835,
                        "num_samples": 1503,
                    },
                    "urd_Arab-mal_Mlym": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 61.24151696606786,
                        "num_samples": 1503,
                    },
                    "urd_Arab-mar_Deva": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 54.52761144377911,
                        "num_samples": 1503,
                    },
                    "urd_Arab-mni_Mtei": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 50.91417165668663,
                        "num_samples": 1503,
                    },
                    "urd_Arab-npi_Deva": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 53.30272787757818,
                        "num_samples": 1503,
                    },
                    "urd_Arab-ory_Orya": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 55.509647371922824,
                        "num_samples": 1503,
                    },
                    "urd_Arab-pan_Guru": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 52.83366600133067,
                        "num_samples": 1503,
                    },
                    "urd_Arab-san_Deva": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 51.4311377245509,
                        "num_samples": 1503,
                    },
                    "urd_Arab-sat_Olck": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 58.94011976047904,
                        "num_samples": 1503,
                    },
                    "urd_Arab-snd_Deva": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 54.445109780439125,
                        "num_samples": 1503,
                    },
                    "urd_Arab-tam_Taml": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 62.590818363273456,
                        "num_samples": 1503,
                    },
                    "urd_Arab-tel_Telu": {
                        "average_sentence1_length": 53.568196939454424,
                        "average_sentence2_length": 51.16300731869594,
                        "num_samples": 1503,
                    },
                },
            }
        },
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata_dict["dataset"])
        self.data_loaded = True
