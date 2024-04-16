from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, CrosslingualTask

# Mapping from ISO 639-3 + script to ISO 639-1 used in TED
_ISO6393_to_ISO6391 = {
    "ara_Arab": "ar",
    "aze_Latn": "az",
    "bel_Cyrl": "be",
    "bul_Cyrl": "bg",
    # "ben_Beng": "bn",
    # "bos_Latn": "bs",
    # "ces_Latn": "cs",
    # "dan_Latn": "da",
    # "deu_Latn": "de",
    # "ell_Grek": "el",
    # "eng_Latn": "en",
    # "epo_Latn": "eo",
    # "spa_Latn": "es",
    # "est_Latn": "et",
    # "eus_Latn": "eu",
    # "fas_Arab": "fa", # this seems new language
    # "fin_Latn": "fi",
    # "fra_Latn": "fr",
    # "glg_Latn": "gl",
    # "heb_Hebr": "he",
    # "hin_Deva": "hi",
    # "hrv_Latn": "hr",
    # "hun_Latn": "hu",
    # "hye_Armn": "hy",
    # "ind_Latn": "id",
    # "ita_Latn": "it",
    # "jpn_Jpan": "ja",
    # "kat_Geor": "ka",
    # "kaz_Cyrl": "kk",
    # "kor_Hang": "ko",
    # "kur_Arab": "ku", # this seems new language
    # "lit_Latn": "lt",
    # "mkd_Cyrl": "mk",
    # "mon_Cyrl": "mn", # this seems new language
    # "mar_Deva": "mr",
    # "msa_Latn": "ms", # this seems new language
    # "mya_Mymr": "my",
    # "nob_Latn": "nb",
    # "nld_Latn": "nl",
    # "pol_Latn": "pl",
    # "por_Latn": "pt",
    # "ron_Latn": "ro",
    # "rus_Cyrl": "ru",
    # "slk_Latn": "sk",
    # "slv_Latn": "sl",
    # "sqi_Latn": "sq", # this seems new language
    # "srp_Cyrl": "sr",
    # "swe_Latn": "sv",
    # "tam_Taml": "ta",
    # "tha_Thai": "th",
    # "tur_Latn": "tr",
    # "ukr_Cyrl": "uk",
    # "urd_Arab": "ur",
    # "vie_Latn": "vi",
    # "zho_Hans": "zh-cn", # new dialect
    # "zho_Hant": "zh-tw", # new dialect
}
_ISO6391_to_ISO6393 = {v: k for k, v in _ISO6393_to_ISO6391.items()}

_SPLIT = ["train"]

# number of sentences to use for evaluation
_N = 256


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


_ISO6393_to_ISO6391_MAPPING = extend_lang_pairs()


class TedTalksBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="TedTalksBitextMining",
        dataset={
            "path": "davidstap/ted_talks",
            "revision": "e840d90795ba7f40dfad0a51ecb6e244840ebdd5",
            "trust_remote_code": True,
        },
        description="Collection of multitarget bitexts based on TED Talks",
        reference="https://huggingface.co/datasets/davidstap/ted_talks",
        type="BitextMining",
        category="s2s",
        eval_splits=_SPLIT,
        eval_langs=_ISO6393_to_ISO6391_MAPPING,
        main_score="f1",
        date=None,
        form=None,
        domains=["Spoken"],
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        n_samples={"train": _N},
        avg_character_length={},
        bibtex_citation="""
@inproceedings{qi-etal-2018-pre,
    title = "When and Why Are Pre-Trained Word Embeddings Useful for Neural Machine Translation?",
    author = "Qi, Ye  and
      Sachan, Devendra  and
      Felix, Matthieu  and
      Padmanabhan, Sarguna  and
      Neubig, Graham",
    editor = "Walker, Marilyn  and
      Ji, Heng  and
      Stent, Amanda",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-2084",
    doi = "10.18653/v1/N18-2084",
    pages = "529--535",
    abstract = "The performance of Neural Machine Translation (NMT) systems often suffers in low-resource scenarios where sufficiently large-scale parallel corpora cannot be obtained. Pre-trained word embeddings have proven to be invaluable for improving performance in natural language analysis tasks, which often suffer from paucity of data. However, their utility for NMT has not been extensively explored. In this work, we perform five sets of experiments that analyze when we can expect pre-trained word embeddings to help in NMT tasks. We show that such embeddings can be surprisingly effective in some cases {--} providing gains of up to 20 BLEU points in the most favorable setting.",
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
        for lang in self.langs:
            self.metadata_dict["dataset"]["path"] = "davidstap/ted_talks"
            self.dataset[lang] = DatasetDict(
                {
                    "train": load_dataset(
                        name="_".join(
                            [_ISO6393_to_ISO6391[l] for l in lang.split("-")]
                        ),
                        split=f"train[:{_N}]",
                        **self.metadata_dict["dataset"],
                    )
                }
            )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        for lang in self.langs:
            l1, l2 = lang.split("-")
            for split in _SPLIT:
                self.dataset[lang][split] = self.dataset[lang][split].rename_column(
                    _ISO6393_to_ISO6391[l1], "sentence1"
                )
                self.dataset[lang][split] = self.dataset[lang][split].rename_column(
                    _ISO6393_to_ISO6391[l2], "sentence2"
                )
