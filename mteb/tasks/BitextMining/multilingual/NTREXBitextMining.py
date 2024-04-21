from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset

from mteb.abstasks import AbsTaskBitextMining, CrosslingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

# Mapping from ISO 639-3 + script to ISO 639-1 used in NTREX
_ISO6393_to_ISO6391 = {
    "ara_Arab": "ar",
    "deu_Latn": "de",
    "eng_Latn": "en",
    "spa_Latn": "es",
    "fra_Latn": "fr",
    "hin_Deva": "hi",
    "ind_Latn": "id",
    "ita_Latn": "it",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "por_Latn": "pt",
    "rus_Cyrl": "ru",
    "tha_Thai": "th",
    "tur_Latn": "tr",
    "vie_Latn": "vi",
    "zho_Hans": "zh",
}

_SPLIT = ["train"]

# number of sentences to use for evaluation
_N = 1997


def extend_lang_pairs() -> dict[str, list[str]]:
    # add all possible language pairs
    hf_lang_subset2isolang = {}
    for x in _ISO6393_to_ISO6391.keys():
        if "-" not in x:
            for y in _ISO6393_to_ISO6391.keys():
                if x != y:
                    pair = f"{x}-{y}"
                    hf_lang_subset2isolang[pair] = [
                        x.replace("_", "-"),
                        y.replace("_", "-"),
                    ]
    return hf_lang_subset2isolang


_EVAL_LANGS = extend_lang_pairs()


class TedTalksBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="NTREXBitextMining",
        dataset={
            "path": "xianf/NTREX",
            "revision": "6047adbf00ceeab9b543bd0bd779b99866deddc0",
            "trust_remote_code": True,
        },
        description="NTREX is a News Test References for MT Evaluation from English into a total of 128 target languages.",
        reference="https://huggingface.co/datasets/xianf/NTREX",
        type="BitextMining",
        category="s2s",
        eval_splits=_SPLIT,
        eval_langs=_EVAL_LANGS,
        main_score="f1",
        date=("2022-11-01", "2022-11-01"),
        form=["written"],
        domains=["News"],
        task_subtypes=None,
        license="CC-BY-SA-4.0",
        socioeconomic_status="medium",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="created",
        n_samples={"train": _N},
        avg_character_length={"train": 120},
        bibtex_citation="""
@inproceedings{federmann-etal-2022-ntrex,
    title = "{NTREX}-128 {--} News Test References for {MT} Evaluation of 128 Languages",
    author = "Federmann, Christian and Kocmi, Tom and Xin, Ying",
    booktitle = "Proceedings of the First Workshop on Scaling Up Multilingual Evaluation",
    month = "nov",
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.sumeval-1.4",
    pages = "21--24",
}
""",
    )

    def load_data(self, **kwargs: Any) -> None:
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}

        all_data = {
            l: load_dataset(
                name=_ISO6393_to_ISO6391[l],
                split=_SPLIT,
                **self.metadata_dict["dataset"],
            )[0]
            for l in _ISO6393_to_ISO6391.keys()
        }

        for lang in self.langs:
            l1, l2 = lang.split("-")
            l1_data = all_data[l1].rename_column("text", "sentence1")
            l2_data = all_data[l2].rename_column("text", "sentence2")
            assert l1_data.num_rows == l2_data.num_rows
            # Combine languages
            data = l1_data.add_column("sentence2", l2_data["sentence2"])
            self.dataset[lang] = DatasetDict({"train": data})

        self.data_loaded = True
