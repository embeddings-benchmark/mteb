from __future__ import annotations

from datasets import DatasetDict, load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskBitextMining, CrosslingualTask

_LANGUAGES = {
    "kat_Geor": "ka",
    "eng_Latn": "en",
}

_EVAL_LANGS = {
    "kat_Geor-eng_Latn": ["kat-Geor", "eng-Latn"],
    "eng_Latn-kat_Geor": ["eng-Latn", "kat-Geor"],
}
_EVAL_SPLIT = "test"


class TbilisiCityHallBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="TbilisiCityHallBitextMining",
        dataset={
            "path": "jupyterjazz/tbilisi-city-hall-titles",
            "revision": "798bb599140565cca2dab8473035fa167e5ee602",
        },
        description="Parallel news titles from the Tbilisi City Hall website (https://tbilisi.gov.ge/).",
        type="BitextMining",
        category="s2s",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_EVAL_LANGS,
        main_score="f1",
        domains=["News"],
        text_creation="created",
        n_samples={_EVAL_SPLIT: 1820},
        reference="https://huggingface.co/datasets/jupyterjazz/tbilisi-city-hall-titles",
        date=("2024-05-02", "2024-05-03"),
        form=["written"],
        task_subtypes=[],
        license="",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        bibtex_citation="",
        avg_character_length={},
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {}

        for lang in self.langs:
            l1, l2 = lang.split("-")
            dataset = load_dataset(
                self.metadata_dict["dataset"]["path"],
                split=_EVAL_SPLIT,
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata_dict["dataset"]["revision"],
            )
            dataset = dataset.rename_columns(
                {_LANGUAGES[l1]: "sentence1", _LANGUAGES[l2]: "sentence2"}
            )
            self.dataset[lang] = DatasetDict({_EVAL_SPLIT: dataset})

        self.data_loaded = True
